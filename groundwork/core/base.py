"""
Groundwork — Base Health Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Universal RAG dimensions shared across all industries.
Every industry template inherits from this base and extends it
with domain-specific compliance dimensions and HITL requirements.

SE USAGE:
  - Never modify this file for a specific customer
  - Extend it in the relevant industry module instead
  - Add new universal dimensions here only if they apply to ALL industries
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ── HITL Configuration ────────────────────────────────────────────────────────

@dataclass
class HITLReviewer:
    """
    Defines a human reviewer role in the pipeline.

    blocking: True  = system holds response until human reviews
    blocking: False = async review, response delivered but flagged
    """
    role: str
    count: int                    # minimum number of reviewers required
    trigger: str                  # "always" | "on_flag" | "quarterly" | custom
    task: str                     # what the human actually does
    blocking: bool = True         # holds pipeline until reviewed?
    escalation_sla_hours: int = 24  # how long before escalating if no review


@dataclass
class HITLConfig:
    """
    Full HITL configuration for an industry.
    Split into pre-production (gates) and in-production (monitoring).
    """
    pre_production: list[HITLReviewer] = field(default_factory=list)
    in_production:  list[HITLReviewer] = field(default_factory=list)

    def get_blockers(self) -> list[HITLReviewer]:
        """Return all reviewers that block the pipeline."""
        return [r for r in self.pre_production + self.in_production if r.blocking]

    def summarize(self) -> str:
        pre  = len(self.pre_production)
        prod = len(self.in_production)
        blocking = len(self.get_blockers())
        total_humans = sum(r.count for r in self.pre_production)
        return (f"{pre} pre-prod reviewer roles ({total_humans} people minimum), "
                f"{prod} in-prod reviewer roles, "
                f"{blocking} blocking checkpoints")


# ── Universal Dimension Scores ────────────────────────────────────────────────

@dataclass
class UniversalScore:
    """
    Scores that apply to every RAG pipeline regardless of industry.
    Scale: 0-10 per dimension. Overall = weighted average.
    """
    retrieval_quality:   float = 0.0   # right doc fetched for the question?
    abstention_behavior: float = 0.0   # correct "I don't know" behavior?
    hallucination_rate:  float = 0.0   # faithfulness to source docs?
    conflict_rate:       float = 0.0   # do docs contradict each other?
    doc_metadata:        float = 0.0   # versioning, dates, status present?
    monitoring_coverage: float = 0.0   # is production being watched?

    # Weights — can be overridden by industry subclass
    WEIGHTS = {
        "retrieval_quality":   0.20,
        "abstention_behavior": 0.20,
        "hallucination_rate":  0.25,
        "conflict_rate":       0.15,
        "doc_metadata":        0.10,
        "monitoring_coverage": 0.10,
    }

    def overall(self) -> float:
        return sum(
            getattr(self, dim) * weight
            for dim, weight in self.WEIGHTS.items()
        )

    def status(self, score: float) -> str:
        if score >= 8.0:  return "✓  GOOD"
        if score >= 5.0:  return "⚠️  NEEDS WORK"
        return "🔴 CRITICAL"


# ── Base Health Check ─────────────────────────────────────────────────────────

class BaseHealthCheck(ABC):
    """
    Abstract base class for all Groundwork industry health checks.

    Every industry module must implement:
      - industry_name()       → display name
      - regulatory_stack()    → list of applicable regulations
      - hitl_config()         → human-in-the-loop requirements
      - domain_dimensions()   → industry-specific eval dimensions
      - run_eval()            → execute the full health check

    Universal dimensions are evaluated here in the base class.
    Industry subclasses call super().run_eval() then add their own.
    """

    @abstractmethod
    def industry_name(self) -> str:
        pass

    @abstractmethod
    def regulatory_stack(self) -> list[str]:
        pass

    @abstractmethod
    def hitl_config(self) -> HITLConfig:
        pass

    @abstractmethod
    def domain_dimensions(self) -> dict[str, float]:
        """
        Returns industry-specific dimension scores.
        Keys are dimension names, values are 0-10 scores.
        """
        pass

    @abstractmethod
    def run_eval(self, questions: list[dict], docs: list) -> dict:
        """
        Run the full health check. Returns results dict.
        """
        pass

    def print_hitl_requirements(self):
        """Print human-in-the-loop requirements for this industry."""
        config = self.hitl_config()
        print(f"\n{'─'*70}")
        print(f"  HUMAN-IN-THE-LOOP REQUIREMENTS — {self.industry_name()}")
        print(f"{'─'*70}")

        print("\n  PRE-PRODUCTION (blocking — cannot go live without these):")
        for r in config.pre_production:
            block = "🔴 BLOCKING" if r.blocking else "🟡 ADVISORY"
            print(f"    {block} | {r.count}x {r.role}")
            print(f"             When: {r.trigger}")
            print(f"             Task: {r.task}")
            print(f"             SLA:  {r.escalation_sla_hours}h before escalation")

        print("\n  IN-PRODUCTION (ongoing review):")
        for r in config.in_production:
            block = "🔴 BLOCKING" if r.blocking else "🟡 ASYNC"
            print(f"    {block} | {r.count}x {r.role}")
            print(f"             When: {r.trigger}")
            print(f"             Task: {r.task}")

        print(f"\n  SUMMARY: {config.summarize()}")

    def print_regulatory_stack(self):
        """Print applicable regulations for this industry."""
        print(f"\n  REGULATORY STACK — {self.industry_name()}")
        for reg in self.regulatory_stack():
            print(f"    • {reg}")
