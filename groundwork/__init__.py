"""
Groundwork — RAG Pipeline Health Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plug-and-play RAG readiness assessment for enterprise AI deployments.

Usage:
    from groundwork.industries.legal import LegalHealthCheck
    from groundwork.industries.healthcare import HealthcareHealthCheck
    from groundwork.industries.financial import FinancialHealthCheck
    from groundwork.industries.tech import TechHealthCheck

    check = LegalHealthCheck()
    check.run_eval()

Or via CLI:
    python groundwork/run.py --industry legal
    python groundwork/run.py --industry healthcare
    python groundwork/run.py --industry financial
    python groundwork/run.py --industry tech
"""

__version__ = "0.1.0"
__author__  = "Groundwork"
