"""
Groundwork — Financial Services Industry Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scenario: "First National Digital Bank" builds a customer support bot
that answers questions using their internal policy documents.

UNIQUE RISKS:
  1. UNANSWERABLE QUESTIONS   → Model hallucinates instead of saying "I don't know"
  2. NUMERICAL PRECISION      → Small errors in rates/fees = compliance violations
  3. CONTEXT STUFFING         → Dumping all docs in works small, breaks at scale
  4. FAITHFULNESS GAP         → Model is faithful to doc, but doc is outdated

REGULATORY STACK:
  FINRA, SEC, CFPB, OCC, Reg Z, Reg E, BSA/AML

HUMAN-IN-THE-LOOP:
  Every numerical claim is a compliance risk. Conflict detection
  must route to a compliance officer — not just flag and proceed.
"""

import os
import sys
import json
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from groundwork.core.base import BaseHealthCheck, HITLConfig, HITLReviewer

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

ANSWERER_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL    = "claude-sonnet-4-6"


# ── HITL Configuration ────────────────────────────────────────────────────────

class FinancialHITLConfig(HITLConfig):
    def __init__(self):
        super().__init__(
            pre_production=[
                HITLReviewer(
                    role="Compliance Officer",
                    count=2,
                    trigger="always — regulatory alignment before deployment",
                    task="Sign off on regulatory alignment across FINRA, CFPB, OCC. "
                         "Review rate/fee disclosure accuracy against Reg Z requirements.",
                    blocking=True,
                    escalation_sla_hours=48
                ),
                HITLReviewer(
                    role="Risk Officer",
                    count=1,
                    trigger="always — risk assessment required",
                    task="Assess blast radius of numerical hallucination scenarios. "
                         "Define acceptable error thresholds for production.",
                    blocking=True,
                    escalation_sla_hours=48
                ),
                HITLReviewer(
                    role="Licensed Financial Advisor",
                    count=1,
                    trigger="if investment or securities topics in scope",
                    task="Review any investment-related response patterns for "
                         "unauthorized financial advice under SEC regulations.",
                    blocking=True,
                    escalation_sla_hours=24
                ),
            ],
            in_production=[
                HITLReviewer(
                    role="Compliance Officer",
                    count=1,
                    trigger="any policy conflict detected",
                    task="Resolve conflicting policy docs before response is delivered. "
                         "Determine which version is authoritative.",
                    blocking=True,
                    escalation_sla_hours=8
                ),
                HITLReviewer(
                    role="Risk Officer / Audit Team",
                    count=3,
                    trigger="monthly — CFPB exam readiness audit",
                    task="Sample review of 10% of all responses for rate/fee accuracy "
                         "and audit trail completeness.",
                    blocking=False,
                    escalation_sla_hours=720
                ),
            ]
        )


# ── Synthetic Knowledge Base ──────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "title": "Savings Account Interest Rates",
        "content": """First National Digital Bank — Savings Rates (Jan 2025)
Standard Savings: 2.10% APY. Premium Savings (balance > $10,000): 4.75% APY.
Jumbo Savings (balance > $100,000): 5.20% APY. Rates are variable and subject
to change with 30 days notice. Minimum opening deposit is $500."""
    },
    {
        "id": "doc_002",
        "title": "Credit Card Annual Percentage Rates",
        "content": """First National Digital Bank Credit Card APRs:
Classic Card: 19.99%-24.99% variable APR. Rewards Card: 21.49%-27.49% variable APR.
Platinum Card: 17.99%-22.99% variable APR. Cash advance APR: 29.99% for all cards.
Late payment fee: $39. Over-limit fee: $29."""
    },
    {
        "id": "doc_003",
        "title": "Wire Transfer Limits and Fees",
        "content": """Wire Transfers — First National Digital Bank:
Domestic: Maximum $50,000/day personal, $500,000/day business. Fee: $25 outgoing.
Incoming wires: free. International: Maximum $25,000/transaction. Fee: $45 + 1% FX.
Wires after 3:00 PM ET processed next business day."""
    },
    {
        "id": "doc_004",
        "title": "Overdraft Policy",
        "content": """Overdraft Policy — First National Digital Bank:
Fee: $34 per overdraft transaction. Maximum 3 overdraft fees per day.
Overdraft protection via linked savings: $10 transfer fee.
Extended overdraft fee: $7/day if account remains negative for more than
5 consecutive business days. Accounts overdrawn by less than $5: no fee."""
    },
    {
        "id": "doc_005",
        "title": "Account Opening Requirements",
        "content": """Account Opening — First National Digital Bank:
Required: Valid government-issued photo ID, SSN or ITIN, minimum opening
deposit ($25 checking, $500 savings), US mailing address.
Non-US citizens: valid passport + ITIN accepted.
Age requirement: 18+ (or joint account with adult co-owner for minors)."""
    },
]

QUESTIONS = [
    {"q": "What is the APY for a Premium Savings account?",
     "answerable": True,  "expected_key": "4.75%"},
    {"q": "How much does an outgoing domestic wire transfer cost?",
     "answerable": True,  "expected_key": "$25"},
    {"q": "What is the overdraft fee per transaction?",
     "answerable": True,  "expected_key": "$34"},
    {"q": "What is the minimum deposit to open a savings account?",
     "answerable": True,  "expected_key": "$500"},
    {"q": "What is the cash advance APR for all credit cards?",
     "answerable": True,  "expected_key": "29.99%"},
    {"q": "What is the maximum number of overdraft fees charged per day?",
     "answerable": True,  "expected_key": "3"},
    {"q": "What is the interest rate on a First National home equity loan?",
     "answerable": False, "expected_key": None},
    {"q": "Does First National offer cryptocurrency trading?",
     "answerable": False, "expected_key": None},
    {"q": "What is the early termination fee for a CD account?",
     "answerable": False, "expected_key": None},
]


# ── Financial Health Check ────────────────────────────────────────────────────

class FinancialHealthCheck(BaseHealthCheck):

    def industry_name(self) -> str:
        return "Financial Services"

    def regulatory_stack(self) -> list[str]:
        return [
            "FINRA — Financial Industry Regulatory Authority",
            "SEC — Securities and Exchange Commission",
            "CFPB — Consumer Financial Protection Bureau",
            "OCC — Office of the Comptroller of the Currency",
            "Reg Z — Truth in Lending Act",
            "Reg E — Electronic Fund Transfer Act",
            "BSA/AML — Bank Secrecy Act / Anti-Money Laundering",
        ]

    def hitl_config(self) -> HITLConfig:
        return FinancialHITLConfig()

    def domain_dimensions(self) -> dict[str, float]:
        return self._domain_scores

    def _retrieve(self, question: str, top_k: int = 2) -> list[dict]:
        corpus  = [d["content"] for d in KNOWLEDGE_BASE]
        tfidf   = TfidfVectorizer().fit(corpus)
        vecs    = tfidf.transform(corpus)
        q_vec   = tfidf.transform([question])
        scores  = cosine_similarity(q_vec, vecs)[0]
        top_idx = scores.argsort()[-top_k:][::-1]
        return [KNOWLEDGE_BASE[i] for i in top_idx]

    def _answer(self, question: str, context_docs: list[dict]) -> str:
        context = "\n\n".join(
            f"[{d['title']}]\n{d['content']}" for d in context_docs
        )
        prompt = f"""You are a customer support assistant for First National Digital Bank.
Answer the customer's question using ONLY the information in the policy documents below.
If the answer is not in the documents, say exactly:
"I don't have that information in my current policy documents."

Policy Documents:
{context}

Customer Question: {question}"""

        response = client.messages.create(
            model=ANSWERER_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    def _judge(self, question: str, answer: str, context_docs: list[dict],
               answerable: bool) -> dict:
        context = "\n\n".join(
            f"[{d['title']}]\n{d['content']}" for d in context_docs
        )
        prompt = f"""You are an expert evaluating a bank chatbot for hallucination and compliance risk.

Question: {question}
Chatbot Answer: {answer}
Source Documents: {context}
Is this answerable from documents? {"YES" if answerable else "NO"}

Evaluate:
1. faithful: Answer only used information from source documents?
2. hallucinated: Did model make up any facts, numbers, or policies?
3. correct_abstain: If unanswerable, did model correctly say it doesn't know?
4. compliance_risk: high / medium / low

Respond in JSON only:
{{
  "faithful": true or false,
  "hallucinated": true or false,
  "correct_abstain": true or false or null,
  "compliance_risk": "high" or "medium" or "low",
  "score": 0.0 to 1.0,
  "finding": "one sentence summary"
}}"""

        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def run_eval(self, questions=None, docs=None) -> dict:
        questions = questions or QUESTIONS

        print("\n" + "═"*80)
        print("  GROUNDWORK — Financial Services")
        print("  Customer: First National Digital Bank")
        print("  Scenario: Customer support RAG bot over internal policy documents")
        print("  Key watch: Numerical accuracy, unanswerable handling, compliance risk")
        print("═"*80)

        self.print_regulatory_stack()
        self.print_hitl_requirements()
        print()

        results        = []
        hallucinations = 0
        high_risk      = 0
        failed_abstain = 0

        for i, item in enumerate(questions):
            q          = item["q"]
            answerable = item["answerable"]

            docs    = self._retrieve(q)
            ans     = self._answer(q, docs)
            verdict = self._judge(q, ans, docs, answerable)

            if verdict.get("hallucinated"):
                hallucinations += 1
            if verdict.get("compliance_risk") == "high":
                high_risk += 1
            if not answerable and not verdict.get("correct_abstain"):
                failed_abstain += 1

            results.append({**item, "answer": ans,
                            "retrieved_docs": [d["id"] for d in docs],
                            "verdict": verdict})

            tags = []
            if verdict.get("hallucinated"):                    tags.append("⚠️  HALLUCINATED")
            if verdict.get("compliance_risk") == "high":       tags.append("🔴 HIGH RISK")
            if not answerable and not verdict.get("correct_abstain"): tags.append("❌ FAILED TO ABSTAIN")

            label = " | ".join(tags) if tags else "✓ OK"

            print(f"\nQ{i+1}: {q}")
            print(f"  Answer  : {ans[:120]}...")
            print(f"  Finding : {verdict.get('finding', '')}")
            print(f"  Status  : {label}")

        total        = len(questions)
        unanswerable = sum(1 for q in questions if not q["answerable"])

        self._domain_scores = {
            "Numerical accuracy":        round((1 - hallucinations/total) * 10, 1),
            "Compliance risk":           round((1 - high_risk/total) * 10, 1),
            "Abstention accuracy":       round((1 - failed_abstain/max(unanswerable,1)) * 10, 1),
            "Hallucination rate":        round((1 - hallucinations/total) * 10, 1),
        }

        print("\n" + "═"*80)
        print("  RESULTS SUMMARY")
        print("═"*80)
        print(f"  Total questions      : {total}")
        print(f"  Hallucinations       : {hallucinations} / {total}")
        print(f"  High compliance risk : {high_risk}")
        print(f"  Failed to abstain    : {failed_abstain} / {unanswerable}")

        print("\n  DOMAIN DIMENSION SCORES (0-10):")
        for dim, score in self._domain_scores.items():
            bar = "█" * int(score) + "░" * (10 - int(score))
            print(f"  {dim:<30} {bar} {score}")

        print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  APPLIED AI TEAM INSIGHTS — What we'd tell First National Digital Bank
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. UNANSWERABLE QUESTIONS are the #1 production failure mode.
     Fix: Explicit "I don't know" instruction + confidence threshold.

  2. NUMERICAL HALLUCINATION is a compliance liability (Reg Z, CFPB).
     Fix: Post-process to verify every number against source doc.

  3. TF-IDF RETRIEVAL fails on paraphrased questions at scale.
     Fix: Upgrade to embedding-based retrieval.

  4. DOCUMENT VERSIONING is essential for CFPB audit readiness.
     Fix: Add effective_date, expiry_date, status metadata to all docs.

  5. JUDGE-AS-EVALUATOR is expensive at scale.
     Fix: Train lightweight classifier on judge outputs for production monitoring.
""")

        with open("results_financial.json", "w") as f:
            json.dump(results, f, indent=2)
        print("  Full results saved to results_financial.json")
        return {"results": results, "domain_scores": self._domain_scores}


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: export ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)
    FinancialHealthCheck().run_eval()
