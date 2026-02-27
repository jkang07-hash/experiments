"""
Groundwork — Single Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run any industry health check from one command.

Usage:
    python groundwork/run.py --industry legal
    python groundwork/run.py --industry healthcare
    python groundwork/run.py --industry financial
    python groundwork/run.py --industry tech

Adding a new industry:
    1. Create groundwork/industries/your_industry.py
    2. Subclass BaseHealthCheck
    3. Register it in INDUSTRY_MAP below
    That's it — plug and play.
"""

import os
import sys
import argparse

# ── Industry Registry ─────────────────────────────────────────────────────────
# To add a new industry: import it and add one line to this dict.
# Nothing else needs to change.

INDUSTRY_MAP = {
    "legal":      "groundwork.industries.legal.LegalHealthCheck",
    # "healthcare": "groundwork.industries.healthcare.HealthcareHealthCheck",
    # "financial":  "groundwork.industries.financial.FinancialHealthCheck",
    # "tech":       "groundwork.industries.tech.TechHealthCheck",
}


def load_check(industry: str):
    if industry not in INDUSTRY_MAP:
        print(f"ERROR: Unknown industry '{industry}'.")
        print(f"Available: {', '.join(INDUSTRY_MAP.keys())}")
        sys.exit(1)

    module_path, class_name = INDUSTRY_MAP[industry].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Groundwork RAG Health Check")
    parser.add_argument("--industry", required=True,
                        choices=list(INDUSTRY_MAP.keys()),
                        help="Industry to run health check for")
    args = parser.parse_args()

    check = load_check(args.industry)
    check.run_eval()
