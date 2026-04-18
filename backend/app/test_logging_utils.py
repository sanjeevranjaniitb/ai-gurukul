"""Tests for the structured JSON logging utility."""

from __future__ import annotations

import json
import logging

import pytest

from backend.app.logging_utils import (
    JSONFormatter,
    correlation_id_var,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)


class TestCorrelationId:
    """Verify correlation ID context variable behaviour."""

    def test_default_is_none(self):
        token = correlation_id_var.set(None)
        try:
            assert get_correlation_id() is None
        finally:
            correlation_id_var.reset(token)

    def test_set_and_get(self):
        set_correlation_id("req-123")
        try:
            assert get_correlation_id() == "req-123"
        finally:
            correlation_id_var.set(None)

    def test_overwrite(self):
        set_correlation_id("first")
        set_correlation_id("second")
        try:
            assert get_correlation_id() == "second"
        finally:
            correlation_id_var.set(None)


class TestJSONFormatter:
    """Verify the JSON formatter produces valid structured output."""

    def _make_record(self, msg: str = "hello", name: str = "test") -> logging.LogRecord:
        return logging.LogRecord(
            name=name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_output_is_valid_json(self):
        fmt = JSONFormatter()
        record = self._make_record()
        line = fmt.format(record)
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_required_keys_present(self):
        fmt = JSONFormatter()
        record = self._make_record()
        parsed = json.loads(fmt.format(record))
        for key in ("timestamp", "level", "component", "correlation_id", "message"):
            assert key in parsed

    def test_component_matches_logger_name(self):
        fmt = JSONFormatter()
        record = self._make_record(name="pdf_parser")
        parsed = json.loads(fmt.format(record))
        assert parsed["component"] == "pdf_parser"

    def test_message_content(self):
        fmt = JSONFormatter()
        record = self._make_record(msg="chunk processed")
        parsed = json.loads(fmt.format(record))
        assert parsed["message"] == "chunk processed"

    def test_level_field(self):
        fmt = JSONFormatter()
        record = self._make_record()
        record.levelname = "WARNING"
        parsed = json.loads(fmt.format(record))
        assert parsed["level"] == "WARNING"

    def test_correlation_id_included_when_set(self):
        set_correlation_id("corr-456")
        try:
            fmt = JSONFormatter()
            record = self._make_record()
            parsed = json.loads(fmt.format(record))
            assert parsed["correlation_id"] == "corr-456"
        finally:
            correlation_id_var.set(None)

    def test_correlation_id_null_when_unset(self):
        correlation_id_var.set(None)
        fmt = JSONFormatter()
        record = self._make_record()
        parsed = json.loads(fmt.format(record))
        assert parsed["correlation_id"] is None

    def test_timestamp_is_iso_format(self):
        fmt = JSONFormatter()
        record = self._make_record()
        parsed = json.loads(fmt.format(record))
        ts = parsed["timestamp"]
        # Should contain 'T' separator and '+' for UTC offset
        assert "T" in ts


class TestGetLogger:
    """Verify get_logger returns a properly configured logger."""

    def test_returns_logger_instance(self):
        logger = get_logger("test_component_a")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches_component(self):
        logger = get_logger("orchestrator")
        assert logger.name == "orchestrator"

    def test_has_json_formatter(self):
        logger = get_logger("test_component_b")
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_no_duplicate_handlers_on_repeated_calls(self):
        name = "test_component_no_dup"
        # Clear any prior state
        existing = logging.getLogger(name)
        existing.handlers.clear()

        get_logger(name)
        get_logger(name)
        logger = logging.getLogger(name)
        assert len(logger.handlers) == 1

    def test_propagate_is_false(self):
        logger = get_logger("test_component_c")
        assert logger.propagate is False

    def test_custom_level(self):
        logger = get_logger("test_component_d", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_emitted_log_is_json(self, capsys):
        name = "test_emit"
        existing = logging.getLogger(name)
        existing.handlers.clear()

        logger = get_logger(name)
        logger.info("test message")

        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed["message"] == "test message"
        assert parsed["component"] == name
