# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM output regressions through the frontend's postprocessor factory."""

import json

import pytest

import dynamo.frontend.sglang_prepost as prepost
from dynamo._internal import UnifiedChatParser
from dynamo.frontend.sglang_processor import SglangProcessor, _build_dynamo_preproc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]


class CharacterTokenizer:
    chat_template = None

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.fixture
def make_post():
    def build(request, *, force_reasoning=False, reasoning_name="glm45", guided=None):
        tools = prepost.convert_tools(request.get("tools"))
        if guided is None:
            guided = prepost.build_tool_call_guided_decoding(
                request, tool_call_parser_name="glm47", sglang_tools=tools
            )
        tool_parser, reasoning_parser = prepost.create_parsers(
            request,
            tool_call_parser_name="glm47",
            reasoning_parser_name=(
                reasoning_name if request.get("separate_reasoning", True) else None
            ),
            sglang_tools=tools,
            force_reasoning=force_reasoning,
            guided_decoding=guided,
        )
        require_reasoning = prepost._guided_output_requires_reasoning(
            request, force_reasoning, reasoning_name, guided
        )
        preproc = _build_dynamo_preproc(
            request,
            [],
            "test-glm",
            [],
            guided,
            tool_parser,
            reasoning_parser,
            require_reasoning=require_reasoning,
            force_reasoning=force_reasoning,
        )
        processor = SglangProcessor(
            tokenizer=CharacterTokenizer(),
            routed_engine=None,
            tool_call_parser_name="glm47",
            reasoning_parser_name=reasoning_name,
            eos_token_ids=None,
        )
        return processor._build_postprocessor(
            request,
            preproc,
            tool_call_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            named_zero_arg_tool=prepost.named_closed_zero_arg_tool(request),
            prompt_token_ids=[],
        )

    return build


def request_for(choice="auto", schema=None):
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": schema or {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": (
            {"type": "function", "function": {"name": "f"}}
            if choice == "named"
            else choice
        ),
    }


def replay(post, text, chunk_size=None):
    step = chunk_size or len(text)
    choices = []
    for offset in range(0, len(text), step):
        choices.extend(
            post.process_outputs(
                {"token_ids": post.tokenizer.encode(text[offset : offset + step])}
            )
        )
    choices.extend(post.process_outputs({"token_ids": [], "finish_reason": "stop"}))
    return choices


def arguments_for(choices, index=0):
    return "".join(
        call["function"].get("arguments", "")
        for choice in choices
        for call in choice["delta"].get("tool_calls", [])
        if call["index"] == index
    )


@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize("force_reasoning", [False, True])
def test_factory_uses_request_reasoning_state(
    make_post, monkeypatch, choice, force_reasoning
):
    initializations = []
    native = prepost.UnifiedChatParser

    def record(*args, **kwargs):
        initializations.append(kwargs)
        return native(*args, **kwargs)

    monkeypatch.setattr(prepost, "UnifiedChatParser", record)
    request = request_for(choice)
    post = make_post(request, force_reasoning=force_reasoning)
    expected_state = (
        "Reasoning" if force_reasoning else "None" if choice == "auto" else "Response"
    )
    assert initializations[0]["starting_state"] == expected_state
    assert initializations[0]["guided"] is (choice != "auto")
    assert initializations[0]["named_tool"] == ("f" if choice == "named" else None)
    body = (
        "<tool_call>f<arg_key>x</arg_key><arg_value>one</arg_value></tool_call>"
        if choice == "auto"
        else '{"x":"one"}'
        if choice == "named"
        else '[{"name":"f","parameters":{"x":"one"}}]'
    )
    prefix = "checking</think>" if force_reasoning else "<think>checking</think>"
    choices = replay(post, prefix + body)
    deltas = [item["delta"] for item in choices]
    assert "".join(delta.get("reasoning_content", "") for delta in deltas) == (
        "" if expected_state == "Response" else "checking"
    )
    assert "".join(delta.get("content", "") for delta in deltas) == (
        prefix if expected_state == "Response" else ""
    )
    assert json.loads(arguments_for(choices)) == {"x": "one"}


@pytest.mark.parametrize("choice", ["auto", "required", "named"])
def test_factory_preserves_unseparated_reasoning(make_post, monkeypatch, choice):
    initializations = []
    native = prepost.UnifiedChatParser

    def record(*args, **kwargs):
        initializations.append(kwargs)
        return native(*args, **kwargs)

    monkeypatch.setattr(prepost, "UnifiedChatParser", record)
    request = request_for(choice)
    request["separate_reasoning"] = False
    post = make_post(request, force_reasoning=True)
    assert initializations[0]["starting_state"] == "Response"
    body = (
        "<tool_call>f<arg_key>x</arg_key><arg_value>one</arg_value></tool_call>"
        if choice == "auto"
        else '{"x":"one"}'
        if choice == "named"
        else '[{"name":"f","parameters":{"x":"one"}}]'
    )
    prefix = "<think>visible</think>"
    choices = replay(post, prefix + body)
    assert "".join(item["delta"].get("content", "") for item in choices) == prefix
    assert not any(item["delta"].get("reasoning_content") for item in choices)
    assert json.loads(arguments_for(choices)) == {"x": "one"}


def test_factory_without_reasoning_parser_keeps_think_literal(make_post):
    post = make_post(request_for(), reasoning_name=None)
    prefix = "<think>visible</think>"
    choices = replay(
        post,
        prefix
        + "<tool_call>f<arg_key>x</arg_key><arg_value>one</arg_value></tool_call>",
    )
    assert "".join(item["delta"].get("content", "") for item in choices) == prefix
    assert not any(item["delta"].get("reasoning_content") for item in choices)
    assert json.loads(arguments_for(choices)) == {"x": "one"}


@pytest.mark.parametrize("choice", ["required", "named"])
def test_factory_preserves_guided_number_precision(make_post, choice):
    post = make_post(request_for(choice, {"type": "number"}))
    arguments = '{"x":9007199254740993.25}'
    text = (
        arguments if choice == "named" else f'[{{"name":"f","parameters":{arguments}}}]'
    )
    choices = replay(post, text, 1)
    assert arguments_for(choices) == arguments


@pytest.mark.parametrize("chunk_size", [1, None])
def test_factory_recovers_two_calls_with_distinct_ids(make_post, chunk_size):
    post = make_post(request_for(), reasoning_name=None)
    text = "\n\n".join(
        f"<tool_call>f<arg_key>x</arg_key><arg_value>{city}</arg_value></tool_call>"
        for city in ("Paris", "Tokyo")
    )
    choices = replay(post, text, chunk_size)
    starts = [
        call
        for choice in choices
        for call in choice["delta"].get("tool_calls", [])
        if "name" in call["function"]
    ]
    assert [call["index"] for call in starts] == [0, 1]
    assert len({call["id"] for call in starts}) == 2
    assert [json.loads(arguments_for(choices, index)) for index in (0, 1)] == [
        {"x": "Paris"},
        {"x": "Tokyo"},
    ]


@pytest.mark.parametrize("chunk_size", [1, None])
@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        (
            "<think>literal</think>",
            {"type": "string"},
            '{"x":"<think>literal</think>"}',
        ),
        ("null", {"type": "string", "nullable": True}, '{"x":null}'),
        ("9007199254740993.25", {"type": "number"}, '{"x":9007199254740993.25}'),
    ],
)
def test_factory_preserves_typed_arguments(
    make_post, chunk_size, value, schema, expected
):
    post = make_post(request_for(schema=schema))
    choices = replay(
        post,
        f"<tool_call>f<arg_key>x</arg_key><arg_value>{value}</arg_value></tool_call>",
        chunk_size,
    )
    assert arguments_for(choices) == expected
    assert not any(item["delta"].get("reasoning_content") for item in choices)


@pytest.mark.parametrize("chunk_size", [1, None])
def test_factory_keeps_reason_tool_reason_content_order(make_post, chunk_size):
    post = make_post(request_for())
    choices = replay(
        post,
        (
            "<think>A</think><tool_call>f<arg_key>x</arg_key>"
            "<arg_value>one</arg_value></tool_call><think>B</think>done"
        ),
        chunk_size,
    )
    channels = []
    for choice in choices:
        for key in ("reasoning_content", "tool_calls", "content"):
            if choice["delta"].get(key) and (not channels or channels[-1] != key):
                channels.append(key)
    assert channels == [
        "reasoning_content",
        "tool_calls",
        "reasoning_content",
        "content",
    ]
    assert choices[-1]["finish_reason"] == "tool_calls"
    assert sum("role" in item["delta"] for item in choices) == 1


def test_factory_keeps_structural_tag_on_native_sglang_path(make_post):
    post = make_post(
        request_for(),
        guided={
            "structural_tag": {
                "type": "structural_tag",
                "format": {"type": "any_text"},
            }
        },
    )
    assert post._unified_parser is None


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Prototype does not support GLM native constrained argument framing",
)
@pytest.mark.parametrize("chunk_size", [1, None])
def test_constrained_object_preserves_argument_closer_literal(chunk_size):
    parser = UnifiedChatParser(
        json.dumps(
            [
                {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"payload": {"type": "object"}},
                    },
                }
            ]
        ),
        constrained=True,
    )
    text = (
        "<tool_call>f<arg_key>payload</arg_key><arg_value>"
        '{"text":"literal </arg_value> inside"}</arg_value></tool_call>'
    )
    step = chunk_size or len(text)
    events = []
    for offset in range(0, len(text), step):
        events.extend(json.loads(parser.advance(text[offset : offset + step], False)))
    events.extend(json.loads(parser.advance("", True)))
    arguments = "".join(event.get("arguments", "") for event in events)
    assert json.loads(arguments) == {"payload": {"text": "literal </arg_value> inside"}}
