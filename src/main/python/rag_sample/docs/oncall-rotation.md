# On-Call Rotation and Escalation

Owner: Platform Engineering
Document ID: OPS-POL-009
Last reviewed: 2026-02-11

## Rotation shape

Each squad runs a weekly primary and secondary rotation. Handover is
Tuesday at 10:00, not Monday, so that a bad weekend is handed over by the
person who lived through it. The schedule lives in PagerDuty and is the
single source of truth.

Primary responsibilities:

- Acknowledge pages within five minutes, day or night.
- Triage, mitigate, then investigate. Mitigation first, always.
- Keep the incident channel updated at least every fifteen minutes during a
  Sev-1, even if the update is "still investigating".

Secondary picks up anything the primary misses after ten minutes and is the
first person the primary calls for a second pair of eyes.

## Severity levels

| Severity | Definition | Response |
|---|---|---|
| Sev-1 | Customer-facing outage, money movement stopped, or data loss | Page primary and incident commander immediately |
| Sev-2 | Major degradation with a workaround | Page primary |
| Sev-3 | Minor degradation, no customer impact | Ticket, next business day |

## Escalation path

1. Primary on-call for the owning squad.
2. Secondary on-call for the same squad.
3. Squad lead.
4. Incident commander on duty (Sev-1 only).
5. Head of Engineering, then the CTO.

Escalate early. Escalating a Sev-2 that turns out to be trivial is never
held against anyone; sitting alone on a Sev-1 for an hour is.

## Compensation and time off

On-call weeks are compensated at a flat weekly allowance plus an hourly
rate for time worked outside business hours. If you are paged between
23:00 and 06:00 you are entitled to take the following morning off; log it
as `ONCALL_REST` in the People Portal, and it is not deducted from your
vacation balance.

## Incident review

Every Sev-1 and Sev-2 gets a written review within five working days.
Reviews are blameless: we look for the systemic cause, not the person who
typed the command. Action items get owners and dates, or they do not count.
