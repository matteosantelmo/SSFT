from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from verl_compat import install_verl_stubs  # noqa: E402

install_verl_stubs()

from verl.utils.output_formatting import (  # noqa: E402
    MARKDOWN_PARSER,
    USER_PROMPT_ROLE,
    XML_PARSER,
    XML_THINK_PARSER,
    MarkdownOutputFormatter,
    OutputFormatter,
    OutputFormattingConfig,
    XMLOutputFormatter,
    XMLThinkOutputFormatter,
    add_formatting_instruction,
    get_output_formatter,
    parse_formatted_output,
)
import generate  # noqa: E402


def _config(**kwargs) -> OutputFormattingConfig:
    return OutputFormattingConfig(parser=MARKDOWN_PARSER, **kwargs)


def test_disabled_prompt_path_returns_original_messages_unchanged():
    messages = [{"role": "user", "content": "Solve this."}]

    result = add_formatting_instruction(messages, OutputFormattingConfig())

    assert result is messages
    assert result == [{"role": "user", "content": "Solve this."}]


def test_instruction_is_appended_to_a_copy_of_existing_system_prompt():
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Solve this."},
    ]
    config = _config(prompt="Use the requested sections.")

    result = add_formatting_instruction(messages, config)

    assert result[0]["content"] == "Be precise.\n\nUse the requested sections."
    assert messages[0]["content"] == "Be precise."


def test_instruction_adds_a_system_message_when_one_is_missing():
    messages = [{"role": "user", "content": "Solve this."}]
    config = _config(prompt="Use the requested sections.")

    result = add_formatting_instruction(messages, config)

    assert result == [
        {"role": "system", "content": "Use the requested sections."},
        {"role": "user", "content": "Solve this."},
    ]


def test_instruction_can_be_appended_to_the_last_user_message():
    messages = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Solve this."},
    ]
    config = _config(prompt="Use the requested sections.", prompt_role=USER_PROMPT_ROLE)

    result = add_formatting_instruction(messages, config)

    assert result[-1]["content"] == "Solve this.\n\nUse the requested sections."
    assert result[0]["content"] == "Earlier question"
    assert messages[-1]["content"] == "Solve this."


def test_user_prompt_role_requires_a_user_message():
    config = _config(prompt_role=USER_PROMPT_ROLE)

    with pytest.raises(ValueError, match="has no user message"):
        add_formatting_instruction(
            [{"role": "system", "content": "System only"}], config
        )


def test_valid_output_separates_reasoning_and_final_response():
    parsed = parse_formatted_output(
        "### Reasoning\n2 + 2 = 4.\n\n### Response\nThe answer is 4.",
        _config(),
    )

    assert parsed.reasoning == "2 + 2 = 4."
    assert parsed.final_response == "The answer is 4."
    assert parsed.format_valid is True
    assert parsed.outcome == "valid"


@pytest.mark.parametrize(
    ("text", "expected_final"),
    [
        (
            "Preamble\n### Reasoning\nwork\n### Response\nanswer",
            "answer",
        ),
        (
            "### Reasoning\nwork\n### Response\nold\n### Response\nanswer",
            "answer",
        ),
    ],
)
def test_malformed_output_with_final_delimiter_recovers_only_its_suffix(
    text, expected_final
):
    parsed = parse_formatted_output(text, _config())

    assert parsed.final_response == expected_final
    assert parsed.format_valid is False
    assert parsed.outcome == "recovered_final_response"


def test_output_without_final_delimiter_falls_back_to_raw_text():
    text = "An unformatted answer"

    parsed = parse_formatted_output(text, _config())

    assert parsed.reasoning is None
    assert parsed.final_response == ""
    assert parsed.verifier_response == text
    assert parsed.format_valid is False
    assert parsed.outcome == "raw_fallback"


def test_parser_selects_its_own_default_prompt():
    markdown = OutputFormattingConfig(parser=MARKDOWN_PARSER)
    xml = OutputFormattingConfig(parser=XML_PARSER)
    xml_think = OutputFormattingConfig(parser=XML_THINK_PARSER)

    assert "### Reasoning" in markdown.instruction
    assert "### Response" in markdown.instruction
    assert "<reasoning>" in xml.instruction
    assert "<answer>" in xml.instruction
    assert "<think>" in xml_think.instruction
    assert "<answer>" not in xml_think.instruction


@pytest.mark.parametrize(
    ("name", "formatter_type"),
    [
        (MARKDOWN_PARSER, MarkdownOutputFormatter),
        (XML_PARSER, XMLOutputFormatter),
        (XML_THINK_PARSER, XMLThinkOutputFormatter),
    ],
)
def test_registered_formatters_implement_common_interface(name, formatter_type):
    formatter = get_output_formatter(name)

    assert isinstance(formatter, OutputFormatter)
    assert isinstance(formatter, formatter_type)
    assert formatter.name == name
    assert formatter.default_prompt
    assert callable(formatter.parse)


def test_configuration_rejects_unknown_parser():
    config = OutputFormattingConfig(parser="custom")

    with pytest.raises(ValueError, match="unknown output-formatting parser"):
        config.validate()


def test_configuration_rejects_prompt_when_parser_is_disabled():
    config = OutputFormattingConfig(prompt="Unused prompt")

    with pytest.raises(ValueError, match="requires a parser"):
        config.validate()


def test_configuration_rejects_unknown_prompt_role():
    config = _config(prompt_role="developer")

    with pytest.raises(ValueError, match="unknown output-formatting prompt role"):
        config.validate()


def test_valid_xml_output_separates_reasoning_and_answer():
    parsed = parse_formatted_output(
        "<reasoning>\n2 + 2 = 4.\n</reasoning>\n<answer>The answer is 4.</answer>",
        OutputFormattingConfig(parser=XML_PARSER),
    )

    assert parsed.reasoning == "2 + 2 = 4."
    assert parsed.final_response == "The answer is 4."
    assert parsed.format_valid is True
    assert parsed.outcome == "valid"


@pytest.mark.parametrize(
    ("text", "expected_final", "outcome"),
    [
        (
            "preamble<reasoning>work</reasoning><answer>answer</answer>",
            "answer",
            "recovered_answer",
        ),
        (
            "<reasoning>work</reasoning><answer>answer",
            "answer",
            "recovered_unclosed_answer",
        ),
        (
            "<reasoning>work</reasoning><answer>old</answer><answer>answer</answer>",
            "answer",
            "recovered_answer",
        ),
    ],
)
def test_malformed_xml_recovers_the_last_answer(text, expected_final, outcome):
    parsed = parse_formatted_output(text, OutputFormattingConfig(parser=XML_PARSER))

    assert parsed.final_response == expected_final
    assert parsed.format_valid is False
    assert parsed.outcome == outcome


def test_xml_without_answer_opening_tag_falls_back_to_raw_text():
    text = "<reasoning>work</reasoning>unformatted answer"

    parsed = parse_formatted_output(text, OutputFormattingConfig(parser=XML_PARSER))

    assert parsed.final_response == ""
    assert parsed.verifier_response == text
    assert parsed.format_valid is False
    assert parsed.outcome == "raw_fallback"


def test_valid_xml_think_output_uses_entire_suffix_as_answer():
    parsed = parse_formatted_output(
        "<think>2 + 2 = 4.</think>\nThe answer is 4.\nExtra detail.",
        OutputFormattingConfig(parser=XML_THINK_PARSER),
    )

    assert parsed.reasoning == "2 + 2 = 4."
    assert parsed.final_response == "The answer is 4.\nExtra detail."
    assert parsed.format_valid is True
    assert parsed.outcome == "valid"


def test_malformed_xml_think_recovers_suffix_after_last_close():
    parsed = parse_formatted_output(
        "preamble<think>old</think><think>work</think>answer",
        OutputFormattingConfig(parser=XML_THINK_PARSER),
    )

    assert parsed.final_response == "answer"
    assert parsed.format_valid is False
    assert parsed.outcome == "recovered_after_reasoning"


def test_xml_think_without_close_falls_back_to_raw_text():
    text = "<think>unfinished reasoning"

    parsed = parse_formatted_output(
        text, OutputFormattingConfig(parser=XML_THINK_PARSER)
    )

    assert parsed.final_response == ""
    assert parsed.verifier_response == text
    assert parsed.format_valid is False
    assert parsed.outcome == "raw_fallback"


class _FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.messages = None

    async def create(self, **kwargs):
        self.messages = kwargs["messages"]
        message = SimpleNamespace(
            content=self.content, reasoning_content=None, reasoning=None
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(completion_tokens=20),
        )


def _process_one(monkeypatch, content, config):
    completions = _FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    queue = asyncio.Queue()
    verifier_inputs = []

    def fake_verify(data_source, response, ground_truth, extra_info):
        verifier_inputs.append(response)
        return {"score": 1.0}

    monkeypatch.setattr(generate, "verify", fake_verify)
    item = {
        "id": "math:0#0",
        "data_source": "math",
        "ability": "math",
        "repeat_idx": 0,
        "seed": 0,
        "messages": [{"role": "user", "content": "2 + 2?"}],
        "ground_truth": "4",
        "extra_info": None,
    }

    async def invoke():
        with ThreadPoolExecutor(max_workers=1) as pool:
            return await generate.process_item(
                item=item,
                client=client,
                model="model",
                gen_sem=asyncio.Semaphore(1),
                verify_sem=asyncio.Semaphore(1),
                verify_pool=pool,
                queue=queue,
                sampling={},
                extra_body_base={},
                enable_thinking=False,
                output_formatting=config,
            )

    return asyncio.run(invoke()), completions.messages, verifier_inputs


def test_process_item_sends_only_parsed_final_response_to_verifier(monkeypatch):
    content = "### Reasoning\n2 + 2 = 4\n### Response\nThe answer is 4."

    record, request_messages, verifier_inputs = _process_one(
        monkeypatch, content, _config(prompt="Use both sections.")
    )

    assert verifier_inputs == ["The answer is 4."]
    assert record["raw_response"] == content
    assert record["reasoning"] == "2 + 2 = 4"
    assert record["response"] == "The answer is 4."
    assert record["output_format_parser"] == MARKDOWN_PARSER
    assert record["output_format_valid"] is True
    assert request_messages[0] == {
        "role": "system",
        "content": "Use both sections.",
    }


def test_process_item_supports_xml_parser(monkeypatch):
    content = "<reasoning>2 + 2 = 4</reasoning><answer>The answer is 4.</answer>"

    record, _, verifier_inputs = _process_one(
        monkeypatch, content, OutputFormattingConfig(parser=XML_PARSER)
    )

    assert verifier_inputs == ["The answer is 4."]
    assert record["reasoning"] == "2 + 2 = 4"
    assert record["response"] == "The answer is 4."
    assert record["output_format_parser"] == XML_PARSER
    assert record["output_format_valid"] is True


def test_process_item_supports_xml_think_and_user_instruction(monkeypatch):
    content = "<think>2 + 2 = 4</think>The answer is 4."
    config = OutputFormattingConfig(
        parser=XML_THINK_PARSER,
        prompt="Use tagged reasoning.",
        prompt_role=USER_PROMPT_ROLE,
    )

    record, request_messages, verifier_inputs = _process_one(
        monkeypatch, content, config
    )

    assert verifier_inputs == ["The answer is 4."]
    assert record["reasoning"] == "2 + 2 = 4"
    assert record["output_format_parser"] == XML_THINK_PARSER
    assert record["output_format_prompt_role"] == USER_PROMPT_ROLE
    assert request_messages == [
        {"role": "user", "content": "2 + 2?\n\nUse tagged reasoning."}
    ]


def test_process_item_disabled_retains_existing_record_and_verifier_path(monkeypatch):
    record, request_messages, verifier_inputs = _process_one(
        monkeypatch, "The answer is 4.", OutputFormattingConfig()
    )

    assert verifier_inputs == ["The answer is 4."]
    assert request_messages == [{"role": "user", "content": "2 + 2?"}]
    assert record["response"] == "The answer is 4."
    assert record["reasoning"] is None
    assert "raw_response" not in record
    assert "output_format_valid" not in record
