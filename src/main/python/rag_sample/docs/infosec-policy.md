# Information Security Policy

Owner: Security Office
Document ID: ISMS-7
Classification: Internal
Last reviewed: 2026-02-02

## Scope

This policy applies to all employees, contractors and third parties who
process Loan Rangers data. It forms part of our Information Security
Management System (ISMS) and supports our SOC 2 Type II attestation.

## Authentication

- Passwords must be at least 14 characters long.
- Password reuse across systems is prohibited.
- Multi-factor authentication (MFA) is mandatory for all production systems,
  the VPN, the People Portal and the AWS console.
- Hardware security keys (FIDO2) are issued to everyone with production
  access. Software TOTP is permitted only as a backup factor.
- Shared accounts are forbidden. Where a shared identity is unavoidable, it
  must be vaulted in 1Password with access logged.

## Data classification

| Level | Label | Examples | Handling |
|---|---|---|---|
| 1 | Public | Marketing pages | No restriction |
| 2 | Internal | Runbooks, org charts | Company accounts only |
| 3 | Confidential | Contracts, financials | Need-to-know, encrypted at rest |
| 4 | Restricted | PII, KYC documents, card data | Need-to-know, audit logged, no export |

Personally identifiable information (PII) and KYC artefacts are always
Level 4 Restricted. Copying Level 4 data to a personal device, a personal
cloud account or an unapproved AI service is a disciplinary matter.

## Device requirements

Company laptops are managed via MDM with full-disk encryption enforced.
The screen lock timeout is ten minutes. Personal devices may access email
and chat only, and never production systems or Level 3+ data.

## Incident reporting

Report any suspected security incident immediately to
security@loanrangers.example or in the `#sec-incident` channel. The target
acknowledgement time is fifteen minutes during business hours and thirty
minutes out of hours.

Do not attempt to investigate a suspected compromise yourself: preserve the
evidence, disconnect the device from the network if instructed, and wait for
the security duty officer. Deleting logs or files during an incident is
prohibited.

Phishing attempts should be forwarded to phishing@loanrangers.example using
the "Report Phishing" button in the mail client.

## Access reviews

Access to Level 3 and Level 4 systems is reviewed quarterly by the system
owner. Accounts inactive for 45 days are automatically disabled. Leavers
lose all access on their last working day as part of the offboarding
checklist.

## Training

Security awareness training is mandatory within the first week of joining
and annually thereafter. Completion is tracked against ISMS-7 for audit
purposes.
