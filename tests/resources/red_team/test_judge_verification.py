# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for judge-time verification units and their wiring into the
EvaluationJudge judge_score path and the AgentBreakerResult verify loop."""

import json

import pytest
from unittest.mock import MagicMock, patch

from garak.attempt import Attempt, Conversation, Message, Turn
from garak.detectors.agent_breaker import AgentBreakerResult
from garak.generators.test import Lipsum
from garak.resources.red_team.evaluation import EvaluationJudge
from garak.resources.red_team.judge_verification import (
    build_verifier_conversation,
    verify_verdict,
)


class VerifyingJudge(EvaluationJudge):
    name = "mockVerifyingJudge"
    evaluator_token_limit = 1024
    system_prompt_judge = "fake prompt with a {goal}"
    system_prompt_on_topic = "fake prompt with a {goal}"
    judge_verification_passes = 1

    def __init__(self):
        self.evaluation_generator = Lipsum(name="gpt-3.5-turbo")


class TestVerifierConversation:

    def test_appends_verdict_and_instruction(self):
        conv = Conversation(
            [
                Turn(role="system", content=Message(text="judge instructions")),
                Turn(role="user", content=Message(text="judge this exchange")),
            ]
        )
        verifier_conv = build_verifier_conversation(conv, "not a rating", "[[N]]")
        assert [t.role for t in verifier_conv.turns] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert verifier_conv.turns[2].content.text == "not a rating"
        assert "[[N]]" in verifier_conv.turns[3].content.text

    def test_original_conversation_untouched(self):
        conv = Conversation(
            [Turn(role="user", content=Message(text="judge this exchange"))]
        )
        build_verifier_conversation(conv, "verdict", "hint")
        assert len(conv.turns) == 1


class TestVerifyVerdict:

    def test_returns_refined_text(self, mocker):
        generator = mocker.MagicMock()
        generator.generate.return_value = [Message("Rating: [[8]]")]
        conv = Conversation([Turn(role="user", content=Message(text="prompt"))])
        assert (
            verify_verdict(generator, conv, "unparseable", "[[N]]") == "Rating: [[8]]"
        )

    def test_generator_error_returns_none(self, mocker):
        generator = mocker.MagicMock()
        generator.generate.side_effect = RuntimeError("API down")
        conv = Conversation([Turn(role="user", content=Message(text="prompt"))])
        assert verify_verdict(generator, conv, "unparseable", "[[N]]") is None

    def test_empty_response_returns_none(self, mocker):
        generator = mocker.MagicMock()
        generator.generate.return_value = []
        conv = Conversation([Turn(role="user", content=Message(text="prompt"))])
        assert verify_verdict(generator, conv, "unparseable", "[[N]]") is None


class TestEvaluationJudgeVerification:

    def test_unparseable_verdict_recovered_by_verification_unit(self, mocker):
        j = VerifyingJudge()
        mock_generate = mocker.patch.object(
            j.evaluation_generator,
            "generate",
            side_effect=[[Message("No rating response")], [Message("Rating: [[8]]")]],
        )
        res = j.judge_score([Message("fake prompt")], ["fake response"])
        assert res == [8.0]
        assert mock_generate.call_count == 2

    def test_verification_failure_falls_back_to_default_score(self, mocker):
        j = VerifyingJudge()
        mock_generate = mocker.patch.object(
            j.evaluation_generator,
            "generate",
            return_value=[Message("No rating response")],
        )
        res = j.judge_score([Message("fake prompt")], ["fake response"])
        assert res == [1.0]  # historic fail-open default
        assert mock_generate.call_count == 2

    def test_parseable_verdict_skips_verification(self, mocker):
        j = VerifyingJudge()
        mock_generate = mocker.patch.object(
            j.evaluation_generator, "generate", return_value=[Message("Rating: [[3]]")]
        )
        res = j.judge_score([Message("fake prompt")], ["fake response"])
        assert res == [3.0]
        mock_generate.assert_called_once()

    def test_default_passes_disables_verification(self, mocker):
        class PlainJudge(EvaluationJudge):
            name = "mockPlainJudge"
            evaluator_token_limit = 1024
            system_prompt_judge = "fake prompt"
            system_prompt_on_topic = "fake prompt"

            def __init__(self):
                self.evaluation_generator = Lipsum(name="gpt-3.5-turbo")

        j = PlainJudge()
        mock_generate = mocker.patch.object(
            j.evaluation_generator,
            "generate",
            return_value=[Message("No rating response")],
        )
        res = j.judge_score([Message("fake prompt")], ["fake response"])
        assert res == [1.0]
        mock_generate.assert_called_once()


def _agent_breaker_detector():
    """AgentBreakerResult with a mocked evaluation generator, as in
    tests/detectors/test_detectors_agent_breaker.py."""
    mock_generator = MagicMock()
    mock_generator.name = "test-model"
    with patch.object(AgentBreakerResult, "_load_generator"):
        d = AgentBreakerResult.__new__(AgentBreakerResult)
        d.detector_model_type = "nim"
        d.detector_model_name = "meta/llama-3.1-405b-instruct"
        d.detector_model_config = {"max_tokens": 1024, "suppressed_params": ["stop"]}
        d.confidence_cutoff = 0.7
        d.verify_attempts = 1
        d.judge_verification_passes = 1
        d.evaluation_generator = mock_generator
        d._load_verify_prompt()
    return d


def _agent_breaker_attempt():
    a = Attempt(probe_classname="test.Test", prompt=Message(text="attack prompt"))
    a.outputs = [Message("response")]
    a.notes.update(
        {
            "current_target": "t",
            "vulnerability_info": "",
            "current_attack_prompt": "",
        }
    )
    return a


class TestAgentBreakerVerification:

    def test_malformed_verdict_recovered_by_verification_unit(self):
        d = _agent_breaker_detector()
        malformed = MagicMock()
        malformed.text = "NOT JSON AT ALL no braces here"
        refined = MagicMock()
        refined.text = json.dumps(
            {"success": "YES", "confidence": 0.9, "reasoning": "exploited"}
        )
        d.evaluation_generator.generate.side_effect = [[malformed], [refined]]
        assert d.detect(_agent_breaker_attempt()) == [1.0]
        assert d.evaluation_generator.generate.call_count == 2

    def test_unrecoverable_verdict_still_unknown(self):
        d = _agent_breaker_detector()
        malformed = MagicMock()
        malformed.text = "NOT JSON AT ALL no braces here"
        d.evaluation_generator.generate.return_value = [malformed]
        assert d.detect(_agent_breaker_attempt()) == [None]

    def test_default_passes_preserves_retry_only_behavior(self):
        d = _agent_breaker_detector()
        d.judge_verification_passes = 0
        malformed = MagicMock()
        malformed.text = "NOT JSON AT ALL no braces here"
        d.evaluation_generator.generate.return_value = [malformed]
        assert d.detect(_agent_breaker_attempt()) == [None]
        # no verification pass: only the verify_attempts judge calls run
        assert d.evaluation_generator.generate.call_count == 1
