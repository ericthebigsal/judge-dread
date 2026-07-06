import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from judgedread.models import TestCaseResult

DEFAULT_HISTORY_PATH = Path(".eval_history.jsonl")


def render_report(
    suite_name: str,
    results: list[TestCaseResult],
    history_path: Path = DEFAULT_HISTORY_PATH,
) -> float:
    console = Console()
    table = Table(title=suite_name)
    table.add_column("Test ID")
    table.add_column("Status")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Deterministic")
    table.add_column("Judge Score")

    for result in results:
        deterministic_cell = (
            "PASS"
            if result.deterministic.passed
            else "FAIL: " + "; ".join(result.deterministic.failures)
        )
        judge_cell = f"{result.judge_verdict.score}/5" if result.judge_verdict else "-"
        table.add_row(
            result.id,
            "[green]PASS[/]" if result.status == "PASS" else "[red]FAIL[/]",
            f"{result.latency_ms:.0f}",
            str(result.input_tokens + result.output_tokens),
            deterministic_cell,
            judge_cell,
        )

    console.print(table)

    pass_rate = (
        sum(1 for r in results if r.status == "PASS") / len(results) if results else 0.0
    )
    console.print(f"Pass rate: {pass_rate:.0%}")

    _check_regression(pass_rate, results, history_path, console)
    _append_history(suite_name, pass_rate, results, history_path)

    return pass_rate


def _check_regression(
    pass_rate: float,
    results: list[TestCaseResult],
    history_path: Path,
    console: Console,
) -> None:
    if not history_path.exists():
        return
    lines = history_path.read_text().strip().splitlines()
    if not lines:
        return
    last_record = json.loads(lines[-1])
    last_statuses = {r["id"]: r["status"] for r in last_record.get("results", [])}

    flipped = [
        r.id for r in results if last_statuses.get(r.id) == "PASS" and r.status == "FAIL"
    ]
    if flipped:
        console.print(
            f"[bold red]REGRESSION: previously passing tests now failing: "
            f"{', '.join(flipped)}[/]"
        )
        return

    if pass_rate < last_record["pass_rate"]:
        console.print(
            f"[bold red]REGRESSION: pass rate dropped from "
            f"{last_record['pass_rate']:.0%} to {pass_rate:.0%}[/]"
        )


def _append_history(
    suite_name: str, pass_rate: float, results: list[TestCaseResult], history_path: Path
) -> None:
    record = {
        "timestamp": time.time(),
        "suite_name": suite_name,
        "pass_rate": pass_rate,
        "results": [{"id": r.id, "status": r.status} for r in results],
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")
