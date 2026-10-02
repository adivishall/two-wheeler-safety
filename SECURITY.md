# Security policy

This is a research prototype, not an enforcement system. Never deploy it to
issue real fines. There is no support SLA; reports are handled on a best-effort
basis by the maintainer.

## Scope

- The Flask app (`app.py`) and its API: injection (SQL, HTML/script in the
  dashboard), path traversal on `/evidence/<file>`, upload handling, auth and
  API-key handling, rate limiting, secrets in config or logs.
- The SQLite store and evidence directory: integrity of violation records and
  evidence packages, and privacy of stored plate text and images (faces and
  plates of people in frame).
- The `Dockerfile` defaults.

Out of scope: model accuracy (open a normal issue), third-party packages
(report upstream), and denial of service that needs more traffic than one
client can send.

## Reporting

Use GitHub private vulnerability reporting: the repository's **Security** tab →
**Report a vulnerability**. Please do not open a public issue for an
unpatched vulnerability.

Include: the commit or branch, the request or input that triggers it (a short
script or `curl` is ideal), what you expected and what happened, and the impact
as you see it. Do not include real people's plates or images; synthetic data is
enough to reproduce everything in this repository.

The threat model and the controls already in place are in
[docs/SECURITY.md](docs/SECURITY.md).
