# Security Policy

## Supported Surface

Security fixes are prioritized for the `voyager-index` OSS foundation surface:

- public package entrypoints under `voyager_index`
- the reference API under `voyager_index.server` and `deploy/reference-api/`
- local storage, persistence, and collection metadata handling
- exposed kernel wrappers and documented CPU/GPU fallback behavior

## Reporting

If you discover a security issue, do not open a public issue with exploit
details.

Use one of these private reporting paths:

- the repository's GitHub Security Advisory / private vulnerability reporting flow
- the maintainer contact method documented in the repository security settings, if advisory reporting is not available

When reporting, include:

- affected file or API surface
- impact and expected severity
- reproduction steps
- whether the issue affects only local deployments or also shipped artifacts

## Response Goals

- acknowledge receipt promptly
- reproduce and triage the issue
- fix and document the affected release surface
- update release notes when the fix ships

## Current Deployment Posture

The reference server is designed as a local-first service. It is not intended to
be presented as an internet-hardened multi-tenant product without additional
authentication, network controls, and deployment-specific hardening.

Current guarantees and limits:

- collection names are restricted to a single validated path segment
- collection metadata writes are atomic and collection mutations use snapshot rollback
- legacy unsafe local index formats such as pickle-backed HNSW fallback state and monolithic ColPali `.pt` snapshots are rejected
- the reference service intentionally runs with `WORKERS=1`; multi-process coordination is not implemented
- `/health` is a liveness check and `/ready` reports degraded startup state such as sparse-index load failures
