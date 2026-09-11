# Data Retention and Deletion Policy

Owner: Data Governance
Document ID: DRP-2024
Last reviewed: 2026-01-30

## Principle

We keep personal data only as long as there is a lawful basis and a
business need. Retention is enforced automatically where possible, because
a policy that depends on someone remembering to run a script is not a
control.

## Retention schedule

| Data category | Retention period | Basis |
|---|---|---|
| KYC identity documents | 5 years after account closure | AML regulation |
| Transaction and ledger records | 7 years | Accounting and tax law |
| Loan application (declined) | 12 months | Legitimate interest |
| Customer support conversations | 24 months | Legitimate interest |
| Marketing consent records | Life of consent + 3 years | Consent evidence |
| Website analytics | 14 months | Consent |
| Application logs containing PII | 30 days | Operational necessity |
| Application logs without PII | 13 months | Operational necessity |
| CCTV footage | 30 days | Legitimate interest |
| Recruitment records (unsuccessful) | 6 months | Legitimate interest |
| Employee records | 6 years after leaving | Employment law |

## Right to erasure

A GDPR erasure request is handled by the Data Governance team within thirty
calendar days. Data held under a legal retention obligation, chiefly KYC
and ledger records, is out of scope of erasure and the requester is told so
explicitly with the legal basis named.

Erasure requests arrive via privacy@loanrangers.example. Never action an
erasure request that arrives directly in a support chat or by phone without
routing it through Data Governance first; identity must be verified.

## Deletion mechanics

Deletion means the record is removed from the primary store and from all
downstream analytical copies. Backups are not selectively edited; instead
backups age out on a 35-day cycle, and a record deleted today is absent
from all backups after 35 days. This window is disclosed in the privacy
notice.

## Data export and residency

Customer data is stored in the EU (eu-central-1) with a warm standby in
eu-west-1. Exporting Level 4 Restricted data outside the EU requires a
documented transfer mechanism and sign-off from Data Governance and Legal.

## Audit

Data Governance samples ten records per category per quarter to confirm the
retention job actually deleted what it claimed. Findings are tracked
against DRP-2024.
