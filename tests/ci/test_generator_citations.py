import asyncio

import pytest

from components.generator import generator_orchestrator as generator_module


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "answer, expected_numbers",
    [
        ("Grant [1].", [1]),
        ("Grant [1]. Induction [3].", [1, 3]),
        ("Induction [3].", [3]),
        ("Both documents [1, 3].", [1, 3]),
        ("No citations.", []),
    ],
)
def test_generator_returns_cited_sources_with_their_numbers(
    monkeypatch, streaming, answer, expected_numbers
):
    filenames = ["grant.txt", "unused.txt", "induction.txt"]
    documents = [
        {"__all_metadata__": {"filename": filename}}
        for filename in filenames
    ]

    # Supply fixed processed documents instead of performing retrieval.
    monkeypatch.setattr(
        generator_module,
        "process_context",
        lambda *args, **kwargs: ("Fixed test context", documents),
    )

    # Set up only the attributes these methods need.
    generator = generator_module.Generator.__new__(generator_module.Generator)
    generator.context_metadata_fields = ["filename"]
    generator.title_metadata_fields = ["filename"]
    generator.link_metadata_field = "url"
    generator.system_prompt = "Answer using the supplied documents."
    generator.instance_user_instructions = ""
    generator.sources_display = "cited"

    async def fixed_answer(messages):
        return answer

    async def fixed_stream(messages):
        # Split the answer to exercise accumulation across streamed chunks.
        for start in range(0, len(answer), 7):
            yield answer[start:start + 7]

    monkeypatch.setattr(generator, "_call_llm", fixed_answer)
    monkeypatch.setattr(generator, "_call_llm_streaming", fixed_stream)

    if streaming:
        async def collect():
            return [
                event
                async for event in generator.generate_streaming(
                    "Test question", context=[], chatui_format=True
                )
            ]

        events = asyncio.run(collect())
        assert not any(event["event"] == "error" for event in events)

        streamed_text = "".join(
            event["data"] for event in events if event["event"] == "data"
        )
        assert streamed_text == answer

        final_events = [
            event for event in events if event["event"] == "final_answer"
        ]
        assert len(final_events) == 1
        sources = final_events[0]["data"]["webSources"]
        assert events[-1]["event"] == "end"
    else:
        result = asyncio.run(
            generator.generate("Test question", context=[], chatui_format=True)
        )
        assert "error" not in result
        assert result["answer"] == answer
        sources = result["sources"]

    assert sources == [
        {"number": number, "title": filenames[number - 1], "uri": None}
        for number in expected_numbers
    ]


class _StubLLMClient:
    provider = "stub"
    model = "stub"


@pytest.mark.parametrize("value, expected", [(None, "cited"), ("all_retrieved", "all_retrieved"), (" Cited ", "cited")])
def test_sources_display_setting_is_resolved(value, expected):
    kwargs = {} if value is None else {"sources_display": value}
    generator = generator_module.Generator(llm_client=_StubLLMClient(), **kwargs)
    assert generator.sources_display == expected


def test_unknown_sources_display_fails_at_startup():
    with pytest.raises(ValueError, match="SOURCES_DISPLAY"):
        generator_module.Generator(llm_client=_StubLLMClient(), sources_display="cited # comment")


def test_all_retrieved_lists_every_document_with_its_number(monkeypatch):
    documents = [{"__all_metadata__": {"filename": name}} for name in ["a.txt", "b.txt", "c.txt"]]
    monkeypatch.setattr(
        generator_module, "process_context", lambda *args, **kwargs: ("ctx", documents)
    )
    generator = generator_module.Generator(llm_client=_StubLLMClient(), sources_display="all_retrieved")

    async def fixed_answer(messages):
        return "Only the second [2]."

    monkeypatch.setattr(generator, "_call_llm", fixed_answer)
    result = asyncio.run(generator.generate("q", context=[], chatui_format=True))

    assert [(s["number"], s["title"]) for s in result["sources"]] == [
        (1, "a.txt"), (2, "b.txt"), (3, "c.txt")
    ]
