# API Gateway Architecture

Owner: Platform Engineering
Document ID: ARCH-004
Last reviewed: 2026-02-18

## Overview

All external traffic enters through the API gateway, which terminates TLS,
authenticates the caller, enforces rate limits and routes to the internal
services over mTLS. Nothing else is exposed to the internet; internal
services have no public listeners.

```
Client -> CDN -> API Gateway -> [auth] -> service mesh -> service
```

## Required headers

| Header | Purpose | Required |
|---|---|---|
| `Authorization` | Bearer token issued by the auth service | Yes |
| `X-LR-Tenant` | Tenant identifier; scopes every query | Yes |
| `X-LR-Request-Id` | Client-supplied idempotency and trace key | Recommended |
| `X-LR-Client-Version` | Used for deprecation telemetry | Recommended |

A request without `X-LR-Tenant` is rejected with HTTP 400 and the error code
`GW-1002`. A request whose token is valid but whose tenant does not match
the token's tenant claim is rejected with HTTP 403 and `GW-1009`; this is
logged as a potential cross-tenant access attempt and alerts the security
duty officer.

## Rate limits

Limits are per tenant per route, enforced with a sliding window:

- Read endpoints: 600 requests per minute.
- Write endpoints: 120 requests per minute.
- Bulk export: 5 requests per minute.

Exceeding a limit returns HTTP 429 with a `Retry-After` header. Clients are
expected to back off exponentially with jitter. Retrying a 429 immediately
in a tight loop will get the tenant temporarily blocklisted.

## Idempotency

Write endpoints accept an `X-LR-Request-Id`. The gateway stores the response
for 24 hours and replays it for a repeated request id, so a client that
times out can safely retry a payment initiation without double-charging.

## Timeouts

The gateway applies a 30-second upstream timeout, and 5 seconds for the
auth call. A gateway timeout surfaces as HTTP 504 with `GW-1504`. Long
running work is not done inline: the service returns 202 with a job id and
the client polls.

## Versioning and deprecation

The API is versioned in the path (`/v1`, `/v2`). A version is supported for
eighteen months after its successor ships. Deprecation is announced in the
changelog, then via the `Sunset` response header, then enforced.
