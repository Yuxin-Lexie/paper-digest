"""OpenAI-backed structured paper analysis."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen

from .arxiv_client import Paper, PaperAnalysis
from .config import AnalysisConfig, DigestTemplate


class OpenAIAnalysisError(RuntimeError):
    """Raised when OpenAI analysis fails."""

@dataclass(slots=True)
class SemanticRelevance:
    score: int
    reason: str


def judge_papers_relevance_with_openai(
    config: AnalysisConfig,
    papers: list[Paper],
    *,
    research_interests: str,
) -> dict[str, SemanticRelevance]:
    """Judge semantic relevance of a batch of papers."""

    if not papers:
        return {}

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise OpenAIAnalysisError(
            f"analysis API key environment variable {config.api_key_env!r} is not set"
        )

    paper_blocks: list[str] = []

    for index, paper in enumerate(papers, start=1):
        paper_blocks.append(
            f"Paper {index}\n"
            f"Paper ID: {paper.paper_id}\n"
            f"{_build_input(paper)}"
        )

    payload = {
        "model": config.model,
        "instructions": (
            "You are screening a batch of academic papers for a research "
            "literature recommender. "
            "Judge EACH paper independently for substantive semantic relevance "
            "to the research interests. "
            "Keywords are only a broad retrieval net and MUST NOT be treated as "
            "evidence that a paper is relevant. "
            "Judge relevance from the actual research question, constructs, "
            "population, theory, methods, and findings described in the title "
            "and abstract. "
            "Ignore accidental lexical overlap, such as a technical paper "
            "containing 'real-time use' when the research interest is human "
            "time use. "
            "A paper may still be relevant when it uses terminology different "
            "from the retrieval keywords. "
            "Use this scale: "
            "0 = unrelated or accidental lexical overlap; "
            "1 = only tangentially related; "
            "2 = adjacent topic with limited direct relevance; "
            "3 = clearly relevant; "
            "4 = highly relevant; "
            "5 = directly central to the research interest. "
            "Return exactly one result for every supplied Paper ID. "
            "Do not omit papers. Do not invent Paper IDs. "
            f"Write each reason in {config.language}. "
            "Return JSON only."
        ),
        "input": (
            f"Research interests:\n{research_interests}\n\n"
            "Papers to screen:\n\n"
            + "\n\n---\n\n".join(paper_blocks)
        ),
        "max_output_tokens": max(1000, len(papers) * 180),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "semantic_relevance_batch",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "paper_id": {"type": "string"},
                                    "score": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                    },
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "paper_id",
                                    "score",
                                    "reason",
                                ],
                            },
                        }
                    },
                    "required": ["results"],
                },
            }
        },
    }

    if config.reasoning_effort != "none":
        payload["reasoning"] = {"effort": config.reasoning_effort}

    request = Request(
        config.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            raw_payload = response.read()
    except OSError as exc:
        raise OpenAIAnalysisError(
            f"failed to judge relevance for paper batch: {exc}"
        ) from exc

    response_json = _load_response_json(raw_payload)
    response_text = _extract_response_text(response_json)

    try:
        raw_result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise OpenAIAnalysisError(
            "semantic relevance batch response was not valid JSON"
        ) from exc

    if not isinstance(raw_result, dict):
        raise OpenAIAnalysisError(
            "semantic relevance batch payload is invalid"
        )

    raw_results = raw_result.get("results")
    if not isinstance(raw_results, list):
        raise OpenAIAnalysisError(
            "semantic relevance batch results must be an array"
        )

    expected_ids = {paper.paper_id for paper in papers}
    parsed_results: dict[str, SemanticRelevance] = {}

    for item in raw_results:
        if not isinstance(item, dict):
            raise OpenAIAnalysisError(
                "semantic relevance batch item is invalid"
            )

        paper_id = _required_string(
            item.get("paper_id"),
            "semantic_relevance.paper_id",
        )
        score = item.get("score")
        reason = _required_string(
            item.get("reason"),
            "semantic_relevance.reason",
        )

        if paper_id not in expected_ids:
            raise OpenAIAnalysisError(
                f"semantic relevance returned unknown paper ID {paper_id!r}"
            )

        if paper_id in parsed_results:
            raise OpenAIAnalysisError(
                f"semantic relevance returned duplicate paper ID {paper_id!r}"
            )

        if not isinstance(score, int) or not 0 <= score <= 5:
            raise OpenAIAnalysisError(
                "semantic relevance score must be an integer from 0 to 5"
            )

        parsed_results[paper_id] = SemanticRelevance(
            score=score,
            reason=reason,
        )

    missing_ids = expected_ids - set(parsed_results)
    if missing_ids:
        raise OpenAIAnalysisError(
            "semantic relevance response omitted paper IDs: "
            + ", ".join(sorted(missing_ids))
        )

    return parsed_results

def analyze_paper_with_openai(
    config: AnalysisConfig,
    paper: Paper,
    *,
    template: DigestTemplate = "default",
) -> PaperAnalysis:
    """Analyze a single paper with the OpenAI Responses API."""

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise OpenAIAnalysisError(
            f"analysis API key environment variable {config.api_key_env!r} is not set"
        )

    payload = {
        "model": config.model,
        "instructions": _build_instructions(config, template=template),
        "input": _build_input(paper),
        "max_output_tokens": config.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "paper_analysis",
                "strict": True,
                "schema": _analysis_schema(),
            }
        },
    }
    if config.reasoning_effort != "none":
        payload["reasoning"] = {"effort": config.reasoning_effort}

    request = Request(
        config.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            raw_payload = response.read()
    except OSError as exc:
        raise OpenAIAnalysisError(
            f"failed to analyze paper {paper.paper_id!r}: {exc}"
        ) from exc

    response_json = _load_response_json(raw_payload)
    response_text = _extract_response_text(response_json)

    try:
        raw_analysis = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise OpenAIAnalysisError(
            "OpenAI analysis response was not valid JSON"
        ) from exc

    return _parse_paper_analysis(raw_analysis)


def _build_instructions(
    config: AnalysisConfig,
    *,
    template: DigestTemplate,
) -> str:
    template_hint = ""
    if template == "zh_daily_brief":
        template_hint = (
            " Prefer newsroom-style phrasing that reads naturally in a Chinese daily"
            " research briefing."
        )

    return (
        "You are writing concise research-digest notes. "
        "Use only the provided title, metadata, and abstract. "
        "Do not invent empirical claims or missing details. "
        "If the abstract does not support a point, say so cautiously. "
        f"Write every field in {config.language}. "
        "Keep each field compact and useful for a daily paper digest. "
        "Return JSON only. Do not use Markdown code fences, headings, "
        "explanatory text, or any text outside the JSON object. "
        "The JSON object must contain exactly these fields: "
        "conclusion, contributions, audience, limitations."
        f"{template_hint}"
    )


def _build_input(paper: Paper) -> str:
    authors = ", ".join(paper.authors) if paper.authors else "Unknown authors"
    categories = (
        ", ".join(paper.categories) if paper.categories else "Unknown categories"
    )
    return (
        f"Title: {paper.title}\n"
        f"Source: {paper.source}\n"
        f"Authors: {authors}\n"
        f"Categories: {categories}\n"
        f"Published: {paper.published_at.isoformat()}\n"
        f"Abstract URL: {paper.abstract_url}\n"
        f"Abstract:\n{paper.summary}"
    )


def _analysis_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "conclusion": {"type": "string"},
            "contributions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "audience": {"type": "string"},
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "conclusion",
            "contributions",
            "audience",
            "limitations",
        ],
    }


def _load_response_json(payload: bytes) -> dict[str, object]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OpenAIAnalysisError("received malformed JSON from OpenAI") from exc

    if not isinstance(raw, dict):
        raise OpenAIAnalysisError("OpenAI response payload is invalid")

    error = raw.get("error")
    if isinstance(error, dict):
        message = error.get("message", "unknown error")
        raise OpenAIAnalysisError(f"OpenAI returned an error: {message}")

    status = raw.get("status")
    if isinstance(status, str) and status not in {"completed", "in_progress"}:
        raise OpenAIAnalysisError(
            f"OpenAI response did not complete successfully: {status}"
        )
    return raw


def _extract_response_text(raw: dict[str, object]) -> str:
    output_text = raw.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = raw.get("output")
    if not isinstance(output, list):
        raise OpenAIAnalysisError("OpenAI response did not include output content")

    fragments: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "refusal":
            raise OpenAIAnalysisError("OpenAI refused to analyze the paper")

        content = item.get("content")
        if not isinstance(content, list):
            continue

        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            item_type = content_item.get("type")
            if item_type in {"output_text", "text"}:
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
            if item_type == "refusal":
                raise OpenAIAnalysisError("OpenAI refused to analyze the paper")

    if not fragments:
        raise OpenAIAnalysisError("OpenAI response did not include analysis text")
    return "\n".join(fragments)


def _parse_paper_analysis(raw: object) -> PaperAnalysis:
    if not isinstance(raw, dict):
        raise OpenAIAnalysisError("OpenAI analysis payload is invalid")

    conclusion = _required_string(raw.get("conclusion"), "analysis.conclusion")
    audience = _required_string(raw.get("audience"), "analysis.audience")
    contributions = _string_list(raw.get("contributions"), "analysis.contributions")
    limitations = _string_list(raw.get("limitations"), "analysis.limitations")

    return PaperAnalysis(
        conclusion=conclusion,
        contributions=contributions,
        audience=audience,
        limitations=limitations,
    )


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise OpenAIAnalysisError(f"{field_name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise OpenAIAnalysisError(f"{field_name} must not be empty")
    return normalized


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise OpenAIAnalysisError(f"{field_name} must be an array of strings")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise OpenAIAnalysisError(f"{field_name} must contain only strings")
        normalized = " ".join(item.split())
        if normalized:
            result.append(normalized)
    return result
