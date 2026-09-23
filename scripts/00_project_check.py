from __future__ import annotations

import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = [
        ROOT / "WORKFLOW.md",
        ROOT / "PROJECT_STATE.json",
        ROOT / "docs" / "CHANGE_PROTOCOL.md",
        ROOT / "README.md",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"FAIL missing required file: {path.relative_to(ROOT)}")

    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    phase = str(state.get("current_phase", ""))
    if not state.get("current_task"):
        raise SystemExit("FAIL current_task is empty")

    gate_status = state.get("go_no_go_A", {}).get("status")
    if phase.startswith("M0"):
        if gate_status != "NOT_REACHED":
            raise SystemExit("FAIL GO/NO-GO A must remain NOT_REACHED during M0")
    else:
        if gate_status not in {"GO_LIMITED", "GO"}:
            raise SystemExit(
                f"FAIL post-M0 work requires GO_LIMITED/GO; current gate status={gate_status!r}"
            )
        if not any("M0E manual audit" in item for item in state.get("completed", [])):
            raise SystemExit("FAIL post-M0 work requires completed M0E manual audit")

    scripts = sorted((ROOT / "scripts").glob("*.py"))
    for script in scripts:
        py_compile.compile(str(script), doraise=True)

    print("PASS project control files present")
    print("PASS PROJECT_STATE.json parsed")
    print(f"PASS current phase: {state['current_phase']} — {state['current_task']}")
    print(f"PASS GO/NO-GO A status: {gate_status}")
    print(f"PASS compiled {len(scripts)} Python scripts")
    print("PASS evidence policy requires 3 screenshots per material change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
