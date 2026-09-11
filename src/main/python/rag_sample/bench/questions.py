"""Evaluation set for the retrieval benchmark.

Gold labels are at document level: a question is answered correctly if the
retriever surfaces a chunk from a document that actually contains the answer.
Chunk-level labels would be more precise but brittle - move a chunk boundary
and the labels rot - and document level is what the downstream LLM needs.

The three categories exist to make the comparison legible rather than to
inflate an average. "lexical" questions carry a rare token that embeddings
blur; "semantic" questions deliberately share no vocabulary with the source
document; "mixed" is what real users type.
"""

LEXICAL = [
    ("What does error E-5031 mean?", ["docs/deployment-runbook.md"]),
    ("How do I clear a stale migration lock after E-5044?",
     ["docs/deployment-runbook.md"]),
    ("Which form is HRP-204 used for?", ["docs/vacation-policy.md"]),
    ("What is form HRP-101 for?",
     ["docs/onboarding-checklist.md", "docs/expense-policy.md",
      "docs/remote-work-policy.md"]),
    ("What is ISMS-7?", ["docs/infosec-policy.md"]),
    ("What does DRP-2024 cover?", ["docs/data-retention-policy.md"]),
    ("Which header is X-LR-Tenant and what happens without it?",
     ["docs/api-gateway-architecture.md"]),
    ("What is returned for GW-1009?", ["docs/api-gateway-architecture.md"]),
    ("What is the lrctl rollback command?", ["docs/deployment-runbook.md"]),
    ("What happened in INC-2026-0304?",
     ["docs/incident-postmortem-2026-03.md"]),
    ("What is the minimum DSCR for commercial lending?", ["docs/glossary.md"]),
    ("What does OPS-RB-003 say about migrations?",
     ["docs/deployment-runbook.md", "docs/incident-postmortem-2026-03.md"]),
]

SEMANTIC = [
    ("How many paid days off do I get each year?",
     ["docs/vacation-policy.md"]),
    ("Can I take my laptop and live in another country for a while?",
     ["docs/remote-work-policy.md"]),
    ("What happens to my unused holiday if I leave the company?",
     ["docs/vacation-policy.md"]),
    ("Someone might have stolen my credentials, what do I do first?",
     ["docs/infosec-policy.md"]),
    ("How much can I spend on a hotel room in London?",
     ["docs/expense-policy.md"]),
    ("Am I allowed to work from a cafe?",
     ["docs/remote-work-policy.md", "docs/infosec-policy.md"]),
    ("If a customer asks us to delete everything about them, can we?",
     ["docs/data-retention-policy.md"]),
    ("What do I do on my very first morning at the company?",
     ["docs/onboarding-checklist.md"]),
    ("Do I get time back if I was woken up at three in the morning?",
     ["docs/oncall-rotation.md"]),
    ("How big should a pull request be?",
     ["docs/code-review-guidelines.md"]),
]

MIXED = [
    ("How long do we keep KYC documents?",
     ["docs/data-retention-policy.md", "docs/glossary.md"]),
    ("Who has to approve a change to the ledger service?",
     ["docs/code-review-guidelines.md"]),
    ("What are the rate limits on write endpoints?",
     ["docs/api-gateway-architecture.md"]),
    ("Is MFA required for the VPN?",
     ["docs/infosec-policy.md", "docs/onboarding-checklist.md"]),
    ("Why did the disbursements stall in March?",
     ["docs/incident-postmortem-2026-03.md"]),
    ("Can I deploy to production on a Friday afternoon?",
     ["docs/deployment-runbook.md"]),
    ("What is the per diem for a full day of travel?",
     ["docs/expense-policy.md"]),
    ("How quickly must an on-call engineer acknowledge a page?",
     ["docs/oncall-rotation.md"]),
]

CATEGORIES = {
    "lexical": LEXICAL,
    "semantic": SEMANTIC,
    "mixed": MIXED,
}

ALL = [
    {"query": q, "gold": gold, "category": name}
    for name, items in CATEGORIES.items()
    for q, gold in items
]
