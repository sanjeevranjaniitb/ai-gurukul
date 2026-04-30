"""ChatAgent: structured RAG pipeline for ``POST /v1/chat`` and ``POST /v1/chat/stream``.

Default **document assistant** path (``document_assistant_mode``): one retrieval query (no LLM rewrite),
no embedding rerank, tighter caps, bounded synthesis tokens—much lower latency than the full
multi-LLM pipeline. End-to-end time under one second is not guaranteed when a remote LLM generates
the answer; logs mark slow turns.

Structured timing logs use logger ``copilot_iitb.chat_pipeline`` (``session_id`` on each step after
session resolution). Message bodies are never logged—only lengths and counts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.llm import IAnswerSynthesizer
from copilot_iitb.core.interfaces.memory import IMemoryStore
from copilot_iitb.core.interfaces.query_planning import IQueryRewriter, IRetrievalPlanner
from copilot_iitb.core.interfaces.retrieval import IRetriever, RetrievedChunk
from copilot_iitb.core.interfaces.session import ISessionRepository, SessionRecord
from copilot_iitb.domain.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    EpisodicTurn,
    RAGAnswer,
    SourceCitation,
)
from copilot_iitb.infrastructure.security.content_policy import aevaluate_input_policy
from copilot_iitb.infrastructure.security.input_guard import InputGuard

pipeline_logger = logging.getLogger("copilot_iitb.chat_pipeline")


class ChatAgentTrace:
    """Collects per-step durations (ms) for logs and optional ``retrieval_debug``."""

    def __init__(self) -> None:
        self._t_start = time.perf_counter()
        self.steps: list[dict[str, Any]] = []
        self.session_id: str | None = None

    def bind_session(self, session_id: str) -> None:
        self.session_id = session_id

    def now(self) -> float:
        return time.perf_counter()

    def record(self, step: str, t0: float, **extra: Any) -> None:
        dt_ms = (time.perf_counter() - t0) * 1000
        row: dict[str, Any] = {"step": step, "duration_ms": round(dt_ms, 2)}
        for k, v in extra.items():
            if v is not None:
                row[k] = v
        self.steps.append(row)
        pipe_row = {**row, "session_id": self.session_id}
        pipeline_logger.info("chat_pipeline %s", json.dumps(pipe_row, default=str))

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t_start) * 1000

    def summary(self) -> dict[str, Any]:
        return {
            "steps": list(self.steps),
            "total_duration_ms": round(self.total_ms(), 2),
        }


@dataclass(frozen=True, slots=True)
class _PendingSynthesis:
    """RAG context is ready; the synthesizer should run (streaming or batch)."""

    trace: ChatAgentTrace
    session: SessionRecord
    clean: str
    top_for_synthesis: list[RetrievedChunk]
    best_sig: float | None
    recent_dialogue: str
    memory_hints: str
    blocks: list[str]
    labels: list[str]
    t_gen: float
    retrieval_debug: dict[str, Any]


def _doc_id(chunk: RetrievedChunk) -> str:
    v = (chunk.metadata or {}).get("document_id")
    return str(v) if v is not None else ""


def select_chunks_for_synthesis(
    ordered_by_relevance: list[RetrievedChunk],
    k: int,
    *,
    max_per_document: int = 2,
) -> list[RetrievedChunk]:
    """Pick up to ``k`` chunks with a soft cap per ``document_id`` for broader context.

    Fills remaining slots from the global ranking if the diversity cap blocks too many.
    """
    if k <= 0:
        return []
    out: list[RetrievedChunk] = []
    per_doc: dict[str, int] = defaultdict(int)
    for c in ordered_by_relevance:
        if len(out) >= k:
            break
        dk = _doc_id(c)
        if dk and per_doc[dk] >= max_per_document:
            continue
        if dk:
            per_doc[dk] += 1
        out.append(c)
    if len(out) >= k:
        return out
    seen = {x.chunk_id for x in out}
    for c in ordered_by_relevance:
        if len(out) >= k:
            break
        if c.chunk_id in seen:
            continue
        out.append(c)
        seen.add(c.chunk_id)
    return out[:k]


class ChatAgent:
    """Single-turn RAG orchestration for ``/v1/chat`` with structured timing logs."""

    def __init__(
        self,
        settings: Settings,
        sessions: ISessionRepository,
        memory: IMemoryStore,
        retriever: IRetriever,
        synthesizer: IAnswerSynthesizer,
        input_guard: InputGuard,
        query_rewriter: IQueryRewriter,
        retrieval_planner: IRetrievalPlanner,
    ) -> None:
        self._settings = settings
        self._sessions = sessions
        self._memory = memory
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._input_guard = input_guard
        self._query_rewriter = query_rewriter
        self._retrieval_planner = retrieval_planner

    def _effective_merge_cap(self) -> int:
        cap = self._settings.retrieval_merge_cap
        if self._settings.document_assistant_mode:
            return min(cap, 12)
        return cap

    def _effective_synthesis_top_k(self) -> int:
        k = self._settings.retrieval_top_k
        if self._settings.document_assistant_mode:
            return min(k, 5)
        return k

    async def _alog_turn_complete(
        self,
        *,
        session_id: str,
        retrieval_debug: dict[str, Any] | None,
        path: str,
    ) -> None:
        ca = (retrieval_debug or {}).get("chat_agent") or {}
        total_ms = ca.get("total_duration_ms")
        row = {
            "event": "chat_turn_complete",
            "session_id": session_id,
            "path": path,
            "total_duration_ms": total_ms,
        }
        pipeline_logger.info("chat_pipeline %s", json.dumps(row, default=str))
        if (
            isinstance(total_ms, (int, float))
            and total_ms > self._settings.chat_slow_turn_warning_ms
        ):
            pipeline_logger.warning(
                "chat_pipeline %s",
                json.dumps(
                    {
                        "event": "chat_turn_slow",
                        "session_id": session_id,
                        "path": path,
                        "total_duration_ms": total_ms,
                        "threshold_ms": self._settings.chat_slow_turn_warning_ms,
                        "note": (
                            "Remote LLM latency usually dominates end-to-end time. "
                            "`document_assistant_mode` removes extra LLM/embed round-trips before synthesis; "
                            "use `POST /v1/chat/stream`, a faster model, or regional hosting to improve perceived speed."
                        ),
                    },
                    default=str,
                ),
            )

    async def _ais_pure_greeting(self, text: str) -> bool:
        return self._is_pure_greeting(text)

    def _recent_dialogue(self, messages: list[ChatMessage]) -> str:
        lines: list[str] = []
        for m in messages:
            lines.append(f"{m.role.value}: {m.content}")
        return "\n".join(lines)

    def _episodic_summary(self, episodic: list[EpisodicTurn]) -> str:
        if not episodic:
            return ""
        parts = [e.summary.strip() for e in episodic[-4:] if e.summary.strip()]
        return "\n".join(parts)[:1600]

    def _chunk_rank_key(self, c: RetrievedChunk) -> float:
        md = c.metadata or {}
        for key in ("rerank_similarity", "vector_similarity"):
            v = md.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        if c.score is not None and isinstance(c.score, (int, float)):
            return float(c.score)
        return 0.0

    def _merge_chunk_pair(self, a: RetrievedChunk, b: RetrievedChunk) -> RetrievedChunk:
        md = dict(a.metadata or {})
        for key, v in (b.metadata or {}).items():
            if key in ("vector_similarity", "rerank_similarity"):
                va = md.get(key)
                if isinstance(va, (int, float)) and isinstance(v, (int, float)):
                    md[key] = max(float(va), float(v))
                elif isinstance(v, (int, float)):
                    md[key] = float(v)
            elif key not in md:
                md[key] = v
        sa = float(a.score) if a.score is not None else 0.0
        sb = float(b.score) if b.score is not None else 0.0
        score = a.score if sa >= sb else b.score
        return RetrievedChunk(chunk_id=a.chunk_id, text=a.text, score=score, metadata=md)

    def _rrf_merge_ranked_lists(
        self, ranked_lists: list[list[RetrievedChunk]], *, cap: int, k: int = 60
    ) -> list[RetrievedChunk]:
        if not ranked_lists:
            return []
        by_id: dict[str, RetrievedChunk] = {}
        id_rank_lists: list[list[str]] = []
        for lst in ranked_lists:
            ids: list[str] = []
            for c in lst:
                ids.append(c.chunk_id)
                if c.chunk_id not in by_id:
                    by_id[c.chunk_id] = c
                else:
                    by_id[c.chunk_id] = self._merge_chunk_pair(by_id[c.chunk_id], c)
            id_rank_lists.append(ids)
        scores: dict[str, float] = {}
        for ranked in id_rank_lists:
            for rank, cid in enumerate(ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        ordered = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [by_id[i] for i in ordered[:cap] if i in by_id]

    async def _retrieve_merged_queries(
        self, queries: tuple[str, ...], filters: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        if not queries:
            return []
        lists = await asyncio.gather(*[self._retriever.aretrieve(q, filters=filters) for q in queries])
        ranked = [sorted(lst, key=self._chunk_rank_key, reverse=True) for lst in lists]
        return self._rrf_merge_ranked_lists(ranked, cap=self._effective_merge_cap())

    def _best_evidence_signal(self, chunks: list[RetrievedChunk]) -> float | None:
        best: float | None = None
        for c in chunks:
            md = c.metadata or {}
            for key in ("rerank_similarity", "vector_similarity"):
                v = md.get(key)
                if isinstance(v, (int, float)):
                    f = float(v)
                    best = f if best is None else max(best, f)
        if best is not None:
            return best
        sims = [float(c.score) for c in chunks if c.score is not None]
        return max(sims) if sims else None

    def _chunks_to_citations(self, chunks: list[RetrievedChunk]) -> list[SourceCitation]:
        cites: list[SourceCitation] = []
        for c in chunks:
            snippet = c.text.strip().replace("\n", " ")
            snippet = snippet[:320]
            cites.append(
                SourceCitation(
                    chunk_id=c.chunk_id,
                    document_id=str(c.metadata.get("document_id")) if c.metadata.get("document_id") else None,
                    title=str(c.metadata.get("title")) if c.metadata.get("title") else None,
                    snippet=snippet,
                    score=c.score,
                )
            )
        return cites

    def _is_pure_greeting(self, text: str) -> bool:
        if not self._settings.enable_greeting_short_circuit:
            return False
        if len(text) > self._settings.greeting_max_message_chars:
            return False
        pattern = (self._settings.greeting_regex or "").strip()
        if not pattern:
            return False
        try:
            return re.match(pattern, text, flags=re.IGNORECASE | re.UNICODE) is not None
        except re.error:
            return False

    async def _execute_until_chunk_selection(
        self, req: ChatRequest
    ) -> ChatResponse | _PendingSynthesis:
        trace = ChatAgentTrace()
        t = trace.now()
        clean = (await self._input_guard.asanitize_user_message(req.message)).value
        policy = await aevaluate_input_policy(self._settings, clean)
        trace.record(
            "guardrails",
            t,
            message_len=len(req.message),
            sanitized_len=len(clean),
            policy_allowed=policy.allowed,
            block_reason=(policy.block_reason if not policy.allowed else None),
        )

        if not policy.allowed:
            t_block = trace.now()
            answer = RAGAnswer(
                answer=self._settings.guardrail_blocked_response,
                citations=[],
                confidence=0.0,
                insufficient_evidence=True,
                follow_up_question=None,
            )
            if req.session_id:
                session = await self._sessions.aget(req.session_id)
                if session is None:
                    raise ValueError(f"Unknown session_id={req.session_id}")
            else:
                session = await self._sessions.acreate(user_id=req.user_id)
            trace.bind_session(session.session_id)
            await self._sessions.atouch(session.session_id)
            await self._memory.aappend_short_term(
                session.session_id, ChatMessage(role=ChatRole.USER, content=clean)
            )
            await self._memory.aappend_short_term(
                session.session_id, ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer)
            )
            summary = f"User asked: {clean[:180]} | Assistant: {answer.answer[:180]}"
            await self._memory.aappend_episodic(
                session.session_id,
                EpisodicTurn(summary=summary, user_intent=None, created_at=datetime.utcnow()),
            )
            trace.record("persist_turn", t_block, branch="policy_block")
            return ChatResponse(
                session_id=session.session_id,
                result=answer,
                retrieval_debug={
                    "blocked": policy.block_reason,
                    "chat_agent": trace.summary(),
                },
            )

        t_sess = trace.now()
        if req.session_id:
            session = await self._sessions.aget(req.session_id)
            if session is None:
                raise ValueError(f"Unknown session_id={req.session_id}")
        else:
            session = await self._sessions.acreate(user_id=req.user_id)
        trace.bind_session(session.session_id)
        await self._sessions.atouch(session.session_id)
        trace.record(
            "session_resolve",
            t_sess,
            reused_session=req.session_id is not None,
        )

        t_memw = trace.now()
        await self._memory.aappend_short_term(session.session_id, ChatMessage(role=ChatRole.USER, content=clean))
        trace.record("memory_write_user", t_memw)

        t_ctx = trace.now()
        user_id = req.user_id
        if user_id:
            short_term, episodic, hints = await asyncio.gather(
                self._memory.aload_short_term(session.session_id, self._settings.short_term_max_messages),
                self._memory.aload_episodic(session.session_id, limit=12),
                self._memory.aload_long_term_hints(user_id, clean, limit=4),
            )
            n_long_term_hints = len(hints)
            memory_hints = (
                self._settings.memory_hints_prefix + "\n- ".join(hints) if hints else ""
            )
        else:
            short_term, episodic = await asyncio.gather(
                self._memory.aload_short_term(session.session_id, self._settings.short_term_max_messages),
                self._memory.aload_episodic(session.session_id, limit=12),
            )
            n_long_term_hints = 0
            memory_hints = ""
        recent = short_term[-self._settings.llm_context_recent_turns :]
        recent_dialogue = self._recent_dialogue(recent)
        episodic_summary = self._episodic_summary(episodic)
        trace.record(
            "memory_load_context",
            t_ctx,
            short_term_n=len(short_term),
            episodic_n=len(episodic),
            long_term_hints_n=n_long_term_hints,
        )

        if await self._ais_pure_greeting(clean):
            t_g = trace.now()
            answer = RAGAnswer(
                answer=self._settings.greeting_response,
                citations=[],
                confidence=0.95,
                insufficient_evidence=False,
                follow_up_question=None,
            )
            trace.record("greeting_short_circuit", t_g)
            t_persist_g = trace.now()
            await self._memory.aappend_short_term(
                session.session_id, ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer)
            )
            summary = f"User asked: {clean[:180]} | Assistant: {answer.answer[:180]}"
            await self._memory.aappend_episodic(
                session.session_id,
                EpisodicTurn(summary=summary, user_intent="greeting", created_at=datetime.utcnow()),
            )
            trace.record("persist_turn", t_persist_g, branch="greeting")
            return ChatResponse(
                session_id=session.session_id,
                result=answer,
                retrieval_debug={"short_circuit": "greeting", "chat_agent": trace.summary()},
            )

        t_plan = trace.now()
        rewrite = await self._query_rewriter.arewrite(
            user_query=clean,
            recent_dialogue=recent_dialogue,
            episodic_summary=episodic_summary,
        )
        base_queries = rewrite.queries if rewrite.queries else (clean,)
        tried_queries: list[str] = list(base_queries)
        trace.record(
            "retrieval_plan",
            t_plan,
            n_search_queries=len(base_queries),
        )

        t_ret = trace.now()
        chunks = await self._retrieve_merged_queries(base_queries, req.metadata_filters)
        trace.record(
            "retrieve_index",
            t_ret,
            n_chunks=len(chunks),
            n_queries=len(base_queries),
        )

        planner_trace: list[dict[str, Any]] = []
        t_iter = trace.now()
        max_steps = (
            self._settings.reasoning_max_iterations if self._settings.enable_reasoning_retrieval else 0
        )
        planner_rounds = 0
        if max_steps > 0:
            for step in range(max_steps):
                planner_rounds += 1
                excerpts = tuple(c.text[:520].strip() for c in chunks[:12] if c.text.strip())
                decision = await self._retrieval_planner.aplan(
                    user_query=clean,
                    memory_hints=memory_hints,
                    tried_queries=tuple(tried_queries),
                    evidence_excerpts=excerpts,
                )
                planner_trace.append(
                    {
                        "step": step,
                        "chain_of_thought": decision.chain_of_thought[:900],
                        "sufficient": decision.sufficient,
                        "follow_up_search_queries": list(decision.follow_up_search_queries),
                    }
                )
                if decision.sufficient or not decision.follow_up_search_queries:
                    break
                new_queries = tuple(decision.follow_up_search_queries)
                tried_queries.extend(new_queries)
                extra_lists = await asyncio.gather(
                    *[self._retriever.aretrieve(q, filters=req.metadata_filters) for q in new_queries]
                )
                ranked_prev = sorted(chunks, key=self._chunk_rank_key, reverse=True)
                ranked_new = [sorted(lst, key=self._chunk_rank_key, reverse=True) for lst in extra_lists]
                chunks = self._rrf_merge_ranked_lists(
                    [ranked_prev, *ranked_new],
                    cap=self._effective_merge_cap(),
                )
        trace.record(
            "retrieval_iterative",
            t_iter,
            enabled=max_steps > 0,
            planner_rounds=planner_rounds,
            n_chunks_after=len(chunks),
        )

        t_sel = trace.now()
        best_sig = self._best_evidence_signal(chunks)
        branch = (
            "no_chunks"
            if not chunks
            else (
                "low_evidence"
                if best_sig is not None and best_sig < self._settings.min_evidence_similarity
                else "synthesize"
            )
        )
        ordered = sorted(chunks, key=self._chunk_rank_key, reverse=True)
        top_for_synthesis = select_chunks_for_synthesis(
            ordered,
            self._effective_synthesis_top_k(),
            max_per_document=2,
        )
        trace.record(
            "chunk_selection",
            t_sel,
            branch=branch,
            best_evidence_signal=best_sig,
            n_ranked=len(ordered),
            n_selected_for_context=len(top_for_synthesis),
        )

        t_gen = trace.now()
        if not chunks:
            answer = RAGAnswer(
                answer=self._settings.no_context_answer,
                citations=[],
                confidence=0.0,
                insufficient_evidence=True,
                follow_up_question=self._settings.low_evidence_follow_up,
            )
        elif best_sig is not None and best_sig < self._settings.min_evidence_similarity:
            answer = RAGAnswer(
                answer=self._settings.low_evidence_answer,
                citations=self._chunks_to_citations(chunks),
                confidence=float(best_sig),
                insufficient_evidence=True,
                follow_up_question=self._settings.low_evidence_follow_up,
            )
        else:
            labels = [f"E{i}" for i in range(1, len(top_for_synthesis) + 1)]
            blocks = [c.text for c in top_for_synthesis]
            debug = {
                "search_queries": list(base_queries),
                "rewrite_notes": rewrite.raw_model_notes,
                "tried_queries": tried_queries,
                "num_chunks": len(chunks),
                "best_evidence_signal": best_sig,
                "reasoning_trace": planner_trace,
            }
            return _PendingSynthesis(
                trace=trace,
                session=session,
                clean=clean,
                top_for_synthesis=top_for_synthesis,
                best_sig=best_sig,
                recent_dialogue=recent_dialogue,
                memory_hints=memory_hints,
                blocks=blocks,
                labels=labels,
                t_gen=t_gen,
                retrieval_debug=debug,
            )
        trace.record(
            "generate_answer",
            t_gen,
            branch=branch,
            insufficient_evidence=answer.insufficient_evidence,
        )

        t_persist = trace.now()
        await self._memory.aappend_short_term(
            session.session_id,
            ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer),
        )
        summary = f"User asked: {clean[:180]} | Assistant: {answer.answer[:180]}"
        await self._memory.aappend_episodic(
            session.session_id,
            EpisodicTurn(summary=summary, user_intent=None, created_at=datetime.utcnow()),
        )
        trace.record("persist_turn", t_persist)

        debug = {
            "search_queries": list(base_queries),
            "rewrite_notes": rewrite.raw_model_notes,
            "tried_queries": tried_queries,
            "num_chunks": len(chunks),
            "best_evidence_signal": best_sig,
            "reasoning_trace": planner_trace,
            "chat_agent": trace.summary(),
        }
        return ChatResponse(session_id=session.session_id, result=answer, retrieval_debug=debug)

    async def run(self, req: ChatRequest) -> ChatResponse:
        out = await self._execute_until_chunk_selection(req)
        if isinstance(out, ChatResponse):
            await self._alog_turn_complete(
                session_id=out.session_id,
                retrieval_debug=out.retrieval_debug,
                path="batch_pipeline",
            )
            return out
        p = out
        answer = await self._synthesizer.asynthesize(
            user_query=p.clean,
            evidence_blocks=p.blocks,
            evidence_labels=p.labels,
            recent_dialogue=p.recent_dialogue,
            memory_hints=p.memory_hints,
        )
        if not answer.citations:
            answer.citations = self._chunks_to_citations(p.top_for_synthesis)
        if p.best_sig is not None and answer.confidence < float(p.best_sig):
            answer.confidence = float(p.best_sig)
        p.trace.record(
            "generate_answer",
            p.t_gen,
            branch="synthesize",
            insufficient_evidence=answer.insufficient_evidence,
        )
        t_persist = p.trace.now()
        await self._memory.aappend_short_term(
            p.session.session_id,
            ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer),
        )
        summary = f"User asked: {p.clean[:180]} | Assistant: {answer.answer[:180]}"
        await self._memory.aappend_episodic(
            p.session.session_id,
            EpisodicTurn(summary=summary, user_intent=None, created_at=datetime.utcnow()),
        )
        p.trace.record("persist_turn", t_persist)
        p.retrieval_debug["chat_agent"] = p.trace.summary()
        resp = ChatResponse(
            session_id=p.session.session_id, result=answer, retrieval_debug=p.retrieval_debug
        )
        await self._alog_turn_complete(
            session_id=resp.session_id,
            retrieval_debug=resp.retrieval_debug,
            path="batch_synthesize",
        )
        return resp

    async def run_stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-shaped dicts for :meth:`ChatService.handle_chat_stream` (meta → delta* → done)."""
        out = await self._execute_until_chunk_selection(req)
        if isinstance(out, ChatResponse):
            yield {
                "type": "done",
                "session_id": out.session_id,
                "result": out.result.model_dump(),
                "retrieval_debug": out.retrieval_debug,
            }
            await self._alog_turn_complete(
                session_id=out.session_id,
                retrieval_debug=out.retrieval_debug,
                path="stream_pipeline",
            )
            return
        p = out
        yield {
            "type": "meta",
            "session_id": p.session.session_id,
            "retrieval_debug": dict(p.retrieval_debug),
        }
        parts: list[str] = []
        async for delta in self._synthesizer.asynthesize_stream(
            user_query=p.clean,
            evidence_blocks=p.blocks,
            evidence_labels=p.labels,
            recent_dialogue=p.recent_dialogue,
            memory_hints=p.memory_hints,
        ):
            parts.append(delta)
            if delta:
                yield {"type": "delta", "text": delta}
        text = "".join(parts).strip()
        if not text:
            text = self._settings.no_context_answer
        answer = RAGAnswer(
            answer=text,
            citations=self._chunks_to_citations(p.top_for_synthesis),
            confidence=float(p.best_sig) if p.best_sig is not None else 0.75,
            insufficient_evidence=False,
            follow_up_question=None,
        )
        if p.best_sig is not None and answer.confidence < float(p.best_sig):
            answer.confidence = float(p.best_sig)
        p.trace.record(
            "generate_answer",
            p.t_gen,
            branch="synthesize",
            insufficient_evidence=answer.insufficient_evidence,
        )
        t_persist = p.trace.now()
        await self._memory.aappend_short_term(
            p.session.session_id,
            ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer),
        )
        summary = f"User asked: {p.clean[:180]} | Assistant: {answer.answer[:180]}"
        await self._memory.aappend_episodic(
            p.session.session_id,
            EpisodicTurn(summary=summary, user_intent=None, created_at=datetime.utcnow()),
        )
        p.trace.record("persist_turn", t_persist)
        p.retrieval_debug["chat_agent"] = p.trace.summary()
        yield {
            "type": "done",
            "session_id": p.session.session_id,
            "result": answer.model_dump(),
            "retrieval_debug": p.retrieval_debug,
        }
        await self._alog_turn_complete(
            session_id=p.session.session_id,
            retrieval_debug=p.retrieval_debug,
            path="stream_synthesize",
        )
