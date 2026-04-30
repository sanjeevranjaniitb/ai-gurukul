"""Application-layer chat facade for ``POST /v1/chat``.

Orchestration lives in :class:`~copilot_iitb.application.chat_agent.ChatAgent`
(guardrails → retrieval plan → index retrieval → chunk selection → generation)
with structured duration logs; this module keeps the historical ``ChatService``
constructor used by ``api/main.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from copilot_iitb.application.chat_agent import ChatAgent
from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.llm import IAnswerSynthesizer
from copilot_iitb.core.interfaces.memory import IMemoryStore
from copilot_iitb.core.interfaces.query_planning import IQueryRewriter, IRetrievalPlanner
from copilot_iitb.core.interfaces.retrieval import IRetriever
from copilot_iitb.core.interfaces.session import ISessionRepository
from copilot_iitb.domain.models import ChatRequest, ChatResponse
from copilot_iitb.infrastructure.security.input_guard import InputGuard


class ChatService:
    """Delegates each turn to :class:`~copilot_iitb.application.chat_agent.ChatAgent`."""

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
        self._agent = ChatAgent(
            settings=settings,
            sessions=sessions,
            memory=memory,
            retriever=retriever,
            synthesizer=synthesizer,
            input_guard=input_guard,
            query_rewriter=query_rewriter,
            retrieval_planner=retrieval_planner,
        )

    async def handle_chat(self, req: ChatRequest) -> ChatResponse:
        return await self._agent.run(req)

    async def handle_chat_stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        async for event in self._agent.run_stream(req):
            yield event
