import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data" / "convert_rl_eval_to_sft.py"
SPEC = importlib.util.spec_from_file_location("convert_rl_eval_to_sft", SCRIPT_PATH)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(converter)


def test_converter_preserves_mcq_prompt_and_rl_verifier_metadata():
    prompt = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": 'Question\n\nA. first\nB. second\n\nGive your final answer as "Answer: $letter".',
        },
    ]
    extra_info = {
        "apply_chat_template_kwargs": {"enable_thinking": True},
        "choices": ["first", "second"],
    }
    source = {
        "data_source": "mmlu",
        "prompt": prompt,
        "reward_model": {"ground_truth": "B", "style": "rule"},
        "extra_info": extra_info,
    }

    converted = converter.convert_row(source, max_new_tokens=128, code_samples=4)
    messages = json.loads(converted["messages"])
    params = json.loads(converted["rollout_params"])

    assert messages[0] == {"role": "system", "content": {"text": prompt[0]["content"]}}
    assert messages[1] == {
        "role": "user",
        "content": {"parts": [{"type": "text", "text": prompt[1]["content"]}]},
    }
    assert params["data_source"] == "mmlu"
    assert params["ground_truth"] == "B"
    assert params["extra_info"] == extra_info
    assert converted["enable_thinking"] is True


def test_converter_defaults_enable_thinking_to_false():
    source = {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": "What is 1 + 1?"}],
        "reward_model": {"ground_truth": "2", "style": "rule"},
        "extra_info": {},
    }

    converted = converter.convert_row(source, max_new_tokens=128, code_samples=4)

    assert converted["enable_thinking"] is False
