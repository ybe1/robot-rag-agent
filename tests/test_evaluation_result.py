import json
from pathlib import Path


RESULT_PATH = (
    Path(__file__).resolve().parent.parent
    / "evaluation"
    / "results"
    / "structured_multievidence.json"
)


def test_structured_retrieval_result_meets_regression_threshold():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["profile"] == "structured"
    assert result["total_cases"] == 60
    assert result["answerable_cases"] == 55
    assert result["passed_cases"] >= 49
    assert result["hit_rate"] >= 0.89
    assert len(result["results"]) == result["total_cases"]
