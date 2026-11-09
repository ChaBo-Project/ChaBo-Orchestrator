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
