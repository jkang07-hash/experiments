"""
Customer Simulation #1: Financial Services
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scenario: "First National Digital Bank" builds a customer support bot
that answers questions using their internal policy documents.

Pipeline:
  [Question] → [Retriever] → [Answerer: claude-haiku] → [Judge: claude-sonnet] → [Score]

KEY LEARNING AREAS (where real customers get stuck):
  1. Unanswerable questions → model hallucinates instead of saying "I don't know"
  2. Numerical precision   → small errors in rates/fees = compliance violations
  3. Context stuffing      → dumping all docs in = works small, breaks at scale
  4. Faithfulness gap      → model is faithful to doc, but doc is outdated
"""

import os
import json
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

ANSWERER_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL    = "claude-sonnet-4-6"

# ── Synthetic Knowledge Base ──────────────────────────────────────────────────
# Simulates a small bank's internal policy docs.
# ⚠️  CUSTOMER TRAP: Real customers often have 500+ docs. They try to stuff
#     all of them into the context. Works here, explodes in production.

KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "title": "Savings Account Interest Rates",
        "content": """First National Digital Bank offers the following savings account rates
as of January 2025: Standard Savings: 2.10% APY. Premium Savings (balance > $10,000):
4.75% APY. Jumbo Savings (balance > $100,000): 5.20% APY. Rates are variable and
subject to change with 30 days notice. Minimum opening deposit is $500."""
    },
    {
        "id": "doc_002",
        "title": "Credit Card Annual Percentage Rates",
        "content": """First National Digital Bank credit cards carry the following APRs:
Classic Card: 19.99% - 24.99% variable APR based on creditworthiness.
Rewards Card: 21.49% - 27.49% variable APR. Platinum Card: 17.99% - 22.99% variable APR.
Cash advance APR is 29.99% for all cards. Late payment fee: $39. Over-limit fee: $29."""
    },
    {
        "id": "doc_003",
        "title": "Wire Transfer Limits and Fees",
        "content": """Domestic wire transfers: Maximum $50,000 per day for personal accounts,
$500,000 per day for business accounts. Fee: $25 per outgoing wire. Incoming wires: free.
International wire transfers: Maximum $25,000 per transaction. Fee: $45 plus 1% FX markup.
Wires initiated after 3:00 PM ET will be processed the next business day."""
    },
    {
        "id": "doc_004",
        "title": "Overdraft Policy",
        "content": """First National Digital Bank charges $34 per overdraft transaction.
Maximum 3 overdraft fees per day. Overdraft protection available via linked savings account
(transfer fee: $10 per transfer). Extended overdraft fee: $7 per day if account remains
negative for more than 5 consecutive business days. Accounts overdrawn by less than $5
are not charged an overdraft fee."""
    },
    {
        "id": "doc_005",
        "title": "Account Opening Requirements",
        "content": """To open a personal checking or savings account, customers must provide:
Valid government-issued photo ID (passport or driver's license), Social Security Number
or ITIN, minimum opening deposit ($25 for checking, $500 for savings), and a US mailing
address. Non-US citizens may open accounts with a valid passport and ITIN.
Age requirement: 18 years or older (or joint account with adult co-owner for minors)."""
    },
]

# ── Question Set ──────────────────────────────────────────────────────────────
# Mix of: answerable, unanswerable, and numerically sensitive questions.
# ⚠️  CUSTOMER TRAP: Most customers only test "answerable" questions in QA.
#     They don't test what happens when the answer ISN'T in the docs — and
#     that's exactly when hallucination happens in production.

QUESTIONS = [
    # Answerable - should get these right
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

    # Numerically sensitive - model might get close but not exact
    # ⚠️  "Close enough" is NOT acceptable in financial services
    {"q": "What is the maximum number of overdraft fees charged per day?",
     "answerable": True,  "expected_key": "3"},
    {"q": "How many days before an extended overdraft fee kicks in?",
     "answerable": True,  "expected_key": "5"},

    # Unanswerable - answer is NOT in the knowledge base
    # ⚠️  This is where hallucination happens. Watch how the model behaves.
    {"q": "What is the interest rate on a First National home equity loan?",
     "answerable": False, "expected_key": None},
    {"q": "Does First National offer cryptocurrency trading?",
     "answerable": False, "expected_key": None},
    {"q": "What is the early termination fee for a CD account?",
     "answerable": False, "expected_key": None},
]


# ── Step 1: Retriever ─────────────────────────────────────────────────────────
# Simple TF-IDF retriever. Finds the most relevant doc for a given question.
# ⚠️  CUSTOMER TRAP: Many customers skip retrieval entirely and just stuff
#     ALL docs into the context. Fine for 5 docs, terrible for 500.

def retrieve(question: str, top_k: int = 2) -> list[dict]:
    corpus  = [d["content"] for d in KNOWLEDGE_BASE]
    tfidf   = TfidfVectorizer().fit(corpus)
    vecs    = tfidf.transform(corpus)
    q_vec   = tfidf.transform([question])
    scores  = cosine_similarity(q_vec, vecs)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return [KNOWLEDGE_BASE[i] for i in top_idx]


# ── Step 2: Answerer ──────────────────────────────────────────────────────────
# Uses retrieved context to answer. Instructed to say "I don't know" if
# the answer isn't in the context.
# ⚠️  CUSTOMER TRAP: Without explicit "I don't know" instruction, the model
#     will hallucinate a plausible-sounding answer every time.

def answer(question: str, context_docs: list[dict]) -> str:
    context = "\n\n".join(
        f"[{d['title']}]\n{d['content']}" for d in context_docs
    )
    prompt = f"""You are a customer support assistant for First National Digital Bank.
Answer the customer's question using ONLY the information in the policy documents below.
If the answer is not in the documents, say exactly: "I don't have that information in my current policy documents."

Policy Documents:
{context}

Customer Question: {question}"""

    response = client.messages.create(
        model=ANSWERER_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


# ── Step 3: Judge ─────────────────────────────────────────────────────────────
# Claude-as-judge evaluates: was the answer faithful? Did it hallucinate?
# ⚠️  CUSTOMER TRAP: Many teams use only rule-based checks (keyword matching).
#     That misses subtle hallucinations like paraphrased wrong numbers.

def judge(question: str, answer_text: str, context_docs: list[dict], answerable: bool) -> dict:
    context = "\n\n".join(
        f"[{d['title']}]\n{d['content']}" for d in context_docs
    )
    prompt = f"""You are an expert evaluator checking a bank chatbot for hallucination and compliance risk.

Question: {question}
Chatbot Answer: {answer_text}
Source Documents: {context}
Is this question answerable from the documents? {"YES" if answerable else "NO"}

Evaluate on these criteria:
1. faithful: Did the answer only use information from the source documents?
2. hallucinated: Did the model make up any facts, numbers, or policies not in the docs?
3. correct_abstain: If unanswerable, did the model correctly say it doesn't know?
4. compliance_risk: Is there a risk of regulatory/legal harm from this answer? (high/medium/low)

Respond in JSON only:
{{
  "faithful": true or false,
  "hallucinated": true or false,
  "correct_abstain": true or false or null,
  "compliance_risk": "high" or "medium" or "low",
  "score": 0.0 to 1.0,
  "finding": "one sentence summary of the key finding"
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


# ── Orchestrator ──────────────────────────────────────────────────────────────
def run():
    print("\n" + "═"*80)
    print("  SIMULATION #1: Financial Services — First National Digital Bank")
    print("  Scenario: Customer support RAG bot over internal policy documents")
    print("═"*80)

    results          = []
    hallucinations   = 0
    high_risk        = 0
    failed_abstain   = 0

    for i, item in enumerate(QUESTIONS):
        q           = item["q"]
        answerable  = item["answerable"]

        docs        = retrieve(q)
        ans         = answer(q, docs)
        verdict     = judge(q, ans, docs, answerable)

        if verdict.get("hallucinated"):
            hallucinations += 1
        if verdict.get("compliance_risk") == "high":
            high_risk += 1
        if not answerable and not verdict.get("correct_abstain"):
            failed_abstain += 1

        results.append({**item, "answer": ans, "retrieved_docs": [d["id"] for d in docs], "verdict": verdict})

        tag = []
        if verdict.get("hallucinated"):       tag.append("⚠️  HALLUCINATED")
        if verdict.get("compliance_risk") == "high": tag.append("🔴 HIGH RISK")
        if not answerable and not verdict.get("correct_abstain"): tag.append("❌ FAILED TO ABSTAIN")

        label = " | ".join(tag) if tag else "✓ OK"
        print(f"\nQ{i+1}: {q}")
        print(f"  Answer  : {ans[:120]}...")
        print(f"  Finding : {verdict.get('finding', '')}")
        print(f"  Status  : {label}")

    # ── Summary + Applied AI Team Insights ────────────────────────────────────
    total = len(QUESTIONS)
    print("\n" + "═"*80)
    print("  RESULTS SUMMARY")
    print("═"*80)
    print(f"  Total questions      : {total}")
    print(f"  Hallucinations       : {hallucinations} / {total}")
    print(f"  High compliance risk : {high_risk}")
    print(f"  Failed to abstain    : {failed_abstain} / {sum(1 for q in QUESTIONS if not q['answerable'])}")

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  APPLIED AI TEAM INSIGHTS — What we'd tell this customer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. UNANSWERABLE QUESTIONS are the #1 production failure mode.
     Fix: Explicit "I don't know" instruction + confidence threshold before answering.

  2. NUMERICAL HALLUCINATION is a compliance liability.
     Fix: Post-process answers to verify any number against source doc with regex/extraction.

  3. TF-IDF RETRIEVAL fails on paraphrased questions.
     Fix: Upgrade to embedding-based retrieval (e.g. Cohere, OpenAI, or local model).

  4. NO VERSIONING on documents means stale answers.
     Fix: Add doc metadata (effective_date, expiry_date) and filter at retrieval time.

  5. JUDGE-AS-EVALUATOR works for dev/test but is expensive at scale.
     Fix: Train a lightweight classifier on judge outputs for production monitoring.
""")

    with open("results_financial.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Full results saved to results_financial.json")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: export ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)
    run()
