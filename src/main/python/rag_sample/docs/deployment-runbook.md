# Production Deployment Runbook

Owner: Platform Engineering
Document ID: OPS-RB-003
Last reviewed: 2026-02-20

## Prerequisites

- Membership of the `deployers` group in Okta.
- The `lrctl` CLI, version 3.2 or newer. Check with `lrctl version`.
- An approved change ticket for anything touching the ledger service.

## Standard deployment

```
lrctl login --sso
lrctl deploy --env staging --service <name> --tag <git-sha>
lrctl smoke --env staging --service <name>
lrctl deploy --env prod --service <name> --tag <git-sha>
```

Deployments to production are canary by default: 5% of traffic for ten
minutes, then 50% for ten minutes, then full rollout. Watch the deployment
dashboard during the canary window.

## Rollback

```
lrctl rollback --env prod --service <name>
```

Rollback restores the previously running image tag. It does not revert
database migrations. If a migration must be undone, page the on-call
database engineer instead of attempting it yourself.

## Deployment freeze

No production deployments on Fridays after 14:00, during the year-end
freeze (20-31 December), or during an active Sev-1 incident. Emergency
fixes during a freeze require approval from the on-call incident commander.

## Common errors

### E-5031 image tag not found

The requested git SHA has no built image in the registry. The CI build
either failed or has not finished. Check the pipeline, then re-run
`lrctl deploy` once the build is green. Do not retag images by hand.

### E-5044 migration lock held

Another deployment is holding the migration advisory lock. Wait for it to
finish. If the lock is stale (older than fifteen minutes with no running
job), clear it with `lrctl migrate --unlock --env prod` and note the action
in the deployment channel.

### E-5102 canary health check failed

The canary pods failed their readiness probe and traffic was withheld.
The deployment aborts automatically and the previous version keeps serving.
Inspect pod logs with `lrctl logs --env prod --service <name> --canary`.

### E-4010 not authorised

Your Okta session lacks the `deployers` role, or the session has expired.
Re-run `lrctl login --sso`. If the error persists, request access through
the People Portal; access grants are not made in chat.

## Post-deployment checks

1. Error rate below 0.5% for ten minutes.
2. p99 latency within 20% of the pre-deployment baseline.
3. No new entries in the ledger reconciliation alert queue.

Record the deployment in the change log. Unrecorded production changes are
an audit finding under ISMS-7.
