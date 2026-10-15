"""One-time patch for app/streamlit_app.py. Run from the repo root:

    python scripts/apply_dashboard_patch.py

Makes four small changes (a backup is saved as app/streamlit_app.py.bak):
  1. imports the EVT panel and the risk-level helper
  2. shows the EVT return-level panel after the uploaded-history section
  3. shows a risk level and the main drivers under a selected-date prediction
  4. defaults the label thresholds to the 'severe hot-day proxy' (38 C, +3.5 C) and turns on
     lagged departure features so the dashboard trains the same model that was evaluated
It checks every anchor first and changes nothing if any is missing or duplicated.
"""
from pathlib import Path
import sys

PATH = Path("app/streamlit_app.py")

PATCHES = [
    ("imports",
     "from src.train_model import TrainingResult, train_baseline_model\n",
     "from src.train_model import TrainingResult, train_baseline_model\n"
     "from src.warning_logic import risk_level\n"
     "from app.evt_panel import render_return_level_panel\n"),
    ("evt panel",
     "    _render_history(dataset)\n",
     "    _render_history(dataset)\n\n"
     "    st.divider()\n"
     "    render_return_level_panel(dataset)\n"),
    ("risk level",
     '    st.caption(\n        "This is a research proxy result, not an official IMD declaration or "\n',
     "    if (\n"
     "        prediction.positive_class_probability is not None\n"
     "        and result.decision_threshold is not None\n"
     "    ):\n"
     "        level = risk_level(prediction.positive_class_probability, result.decision_threshold)\n"
     '        st.metric("Risk level (research proxy)", level)\n'
     "        st.caption(\n"
     '            f"Probability of a heatwave-proxy day: {prediction.positive_class_probability:.1%}. "\n'
     '            f"Warning threshold {result.decision_threshold:.1%}, chosen to minimise cost "\n'
     '            "(a missed event counted as 10 false alarms) on the first half of the validation "\n'
     '            "period. The cost ratio is an assumption."\n'
     "        )\n"
     "    if prediction.contributing_factors:\n"
     '        st.markdown("**Main drivers of this probability** (larger = pushes towards a warning)")\n'
     "        for name, value in prediction.contributing_factors:\n"
     '            st.markdown(f"- `{name}`: {value:+.2f}")\n'
     '    st.caption(\n        "This is a research proxy result, not an official IMD declaration or "\n'),
    ("default thresholds",
     '        "Proxy Tmax threshold (°C)", value=37.0, step=0.5\n',
     '        "Proxy Tmax threshold (°C)", value=38.0, step=0.5\n'),
    ("default departure",
     '        "Proxy departure threshold (°C)", value=4.5, step=0.5\n',
     '        "Proxy departure threshold (°C)", value=3.5, step=0.5\n'),
    ("latest date first",
     "        options=dates,\n",
     "        options=dates,\n        index=len(dates) - 1,\n"),
    ("departure features",
     '        "train_fraction": float(train_fraction),\n    }\n',
     '        "train_fraction": float(train_fraction),\n        "use_departure_features": True,\n    }\n'),
]


def main() -> int:
    if not PATH.exists():
        print(f"{PATH} not found. Run this from the repository root.")
        return 1
    text = PATH.read_text(encoding="utf-8")
    todo = []
    for name, old, new in PATCHES:
        if new in text:                      # already applied (safe to run again)
            continue
        n = text.count(old)
        if n != 1:
            print(f"Cannot patch: anchor '{name}' found {n} times (expected 1). Nothing was changed.")
            print("Apply the change by hand: see scripts/apply_dashboard_patch.py for the exact text.")
            return 1
        todo.append((name, old, new))
    if not todo:
        print("Already fully patched. Nothing to do.")
        return 0
    Path(str(PATH) + ".bak").write_text(text, encoding="utf-8")
    for _, old, new in todo:
        text = text.replace(old, new)
    PATH.write_text(text, encoding="utf-8")
    print("Applied: " + ", ".join(n for n, _, _ in todo) + " (backup: app/streamlit_app.py.bak).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
