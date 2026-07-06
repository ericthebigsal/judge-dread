import json

from judgedread.models import DeterministicResult, JudgeVerdict, TestCaseResult
from judgedread.report import render_report


def _result(id_, status, judge_score=None):
    verdict = JudgeVerdict(score=judge_score, reasoning="r") if judge_score is not None else None
    return TestCaseResult(
        id=id_,
        description="desc",
        status=status,
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=20,
        deterministic=DeterministicResult(passed=(status == "PASS")),
        judge_verdict=verdict,
        raw_output="{}",
    )


def test_render_report_writes_history_record_with_per_test_statuses(tmp_path, capsys):
    history_path = tmp_path / ".eval_history.jsonl"
    results = [_result("TC_001", "PASS"), _result("TC_002", "FAIL")]

    pass_rate = render_report("Suite", results, history_path=history_path)

    assert pass_rate == 0.5
    lines = history_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["suite_name"] == "Suite"
    assert record["pass_rate"] == 0.5
    assert record["results"] == [
        {"id": "TC_001", "status": "PASS"},
        {"id": "TC_002", "status": "FAIL"},
    ]
    captured = capsys.readouterr()
    assert "TC_001" in captured.out
    assert "TC_002" in captured.out


def test_render_report_appends_to_existing_history(tmp_path):
    history_path = tmp_path / ".eval_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "timestamp": 1,
                "suite_name": "Suite",
                "pass_rate": 1.0,
                "results": [{"id": "TC_001", "status": "PASS"}, {"id": "TC_002", "status": "PASS"}],
            }
        )
        + "\n"
    )
    results = [_result("TC_001", "PASS"), _result("TC_002", "FAIL")]

    render_report("Suite", results, history_path=history_path)

    lines = history_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_render_report_flags_specific_test_that_flipped_to_failing(tmp_path, capsys):
    history_path = tmp_path / ".eval_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "timestamp": 1,
                "suite_name": "Suite",
                "pass_rate": 1.0,
                "results": [{"id": "TC_001", "status": "PASS"}, {"id": "TC_002", "status": "PASS"}],
            }
        )
        + "\n"
    )
    results = [_result("TC_001", "PASS"), _result("TC_002", "FAIL")]

    render_report("Suite", results, history_path=history_path)

    captured = capsys.readouterr()
    assert "REGRESSION" in captured.out
    assert "TC_002" in captured.out
    assert "TC_001" not in captured.out.split("REGRESSION")[1]


def test_render_report_flags_aggregate_drop_when_test_ids_are_new(tmp_path, capsys):
    history_path = tmp_path / ".eval_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "timestamp": 1,
                "suite_name": "Suite",
                "pass_rate": 1.0,
                "results": [{"id": "TC_OLD", "status": "PASS"}],
            }
        )
        + "\n"
    )
    results = [_result("TC_NEW_1", "PASS"), _result("TC_NEW_2", "FAIL")]

    render_report("Suite", results, history_path=history_path)

    captured = capsys.readouterr()
    assert "REGRESSION" in captured.out
    assert "pass rate dropped" in captured.out


def test_render_report_with_no_results_has_zero_pass_rate(tmp_path):
    history_path = tmp_path / ".eval_history.jsonl"
    pass_rate = render_report("Empty Suite", [], history_path=history_path)
    assert pass_rate == 0.0
