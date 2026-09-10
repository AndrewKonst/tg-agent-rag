# Incident Review: Ledger Reconciliation Stall, 2026-03-04

Incident ID: INC-2026-0304
Severity: Sev-1
Duration: 2h 41m (07:12 - 09:53 UTC)
Incident commander: Platform on-call
Status: Closed, action items open

## Customer impact

Disbursements queued but did not settle for two hours and forty-one
minutes. 1,842 loan disbursements were delayed. No money was lost and no
double payments occurred; the idempotency layer in the API gateway held.
Fourteen customers contacted support during the window.

## Timeline (UTC)

| Time | Event |
|---|---|
| 07:12 | Deployment of ledger service tag `9f3ac21` to prod completes |
| 07:19 | Reconciliation queue depth alert fires, acknowledged in 3m |
| 07:26 | Primary escalates to secondary; queue depth still climbing |
| 07:35 | Sev-1 declared, incident commander paged |
| 07:48 | Cause narrowed to the new migration holding a row lock |
| 08:05 | `lrctl rollback --env prod --service ledger` attempted |
| 08:11 | Rollback completes but the stall persists: the migration was not reverted |
| 08:40 | Database engineer paged, long-running transaction identified |
| 09:02 | Blocking transaction terminated manually |
| 09:20 | Queue begins draining; disbursements start settling |
| 09:53 | Queue at baseline, incident closed |

## Root cause

The deployment included a migration that added an index without the
`CONCURRENTLY` option, taking an `ACCESS EXCLUSIVE` lock on the
`ledger_entries` table. The reconciliation worker's write transactions
queued behind that lock. Because the worker retried with a fixed backoff
and no circuit breaker, the queue grew faster than it drained even after
the lock cleared.

Rollback did not help because `lrctl rollback` restores the previous image
tag only and never reverts migrations, exactly as documented in OPS-RB-003.
The on-call engineer expected rollback to be a full undo.

## Contributing factors

- The migration linter did not flag the missing `CONCURRENTLY` because the
  rule only covered `CREATE INDEX` and not `CREATE UNIQUE INDEX`.
- The runbook documents that rollback excludes migrations, but the sentence
  sits at the end of a long section and was not read under pressure.
- The reconciliation worker had no circuit breaker.

## Action items

| # | Action | Owner | Due |
|---|---|---|---|
| 1 | Extend the migration linter to all index creation statements | Platform | 2026-03-18 |
| 2 | Add a circuit breaker and jittered backoff to the reconciliation worker | Ledger squad | 2026-03-25 |
| 3 | Add a rollback pre-flight check that warns when the release contains migrations | Platform | 2026-04-01 |
| 4 | Move the "rollback does not revert migrations" warning to the top of OPS-RB-003 | Platform | 2026-03-11 |

## What went well

Detection was fast: the alert fired seven minutes after the deployment and
was acknowledged in three. The idempotency layer prevented duplicate
disbursements entirely. Communication in the incident channel was steady
and honest about uncertainty.
