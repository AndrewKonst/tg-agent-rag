# Code Review Guidelines

Owner: Engineering Practice
Document ID: ENG-GUIDE-002
Last reviewed: 2026-02-14

## Intent

Review exists to catch defects, spread context and keep the codebase
coherent. It does not exist to prove seniority. A review that makes the
author dread the next one has failed even if it found a bug.

## Author responsibilities

- Keep pull requests under roughly 400 changed lines. Split larger work.
- Write a description that says what changed, why, and how it was verified.
- Never mix a refactor and a behaviour change in one pull request.
- Self-review the diff before requesting a reviewer.
- Include the ticket id in the title.

## Reviewer responsibilities

- First review within one business day. If you cannot, decline explicitly
  so the author can find someone else rather than waiting in silence.
- Distinguish blocking from non-blocking clearly. Prefix optional remarks
  with `nit:` and mean it: a `nit:` never blocks a merge.
- Explain the why behind a request, not just the what.
- Approve when the change is better than what is on the branch today, not
  when it is perfect.

## Required approvals

| Change | Approvals |
|---|---|
| Ordinary service code | 1 |
| Ledger service or money movement | 2, one from the ledger squad |
| Migrations | 2, one from a database engineer |
| Infrastructure or IAM policy | 2, one from Platform |
| Anything touching PII handling | 2, one from Security |

Migrations get extra scrutiny for a reason: see INC-2026-0304, where an
index built without `CONCURRENTLY` stalled disbursements for nearly three
hours.

## Automated gates

Tests, linting, type checks and the migration linter run in CI and must be
green before merge. Do not merge with a failing check and a promise to fix
it after; the branch protection rule will not let you anyway.

## Disagreement

If author and reviewer cannot agree within two rounds, bring in a third
engineer rather than continuing in comments. Escalate to the squad lead if
that does not settle it. Decide, record the decision in the pull request,
and move on.
