# Security Policy

## Reporting a Vulnerability

Please do not report suspected security vulnerabilities in public GitHub issues, discussions, or Discord.

Use GitHub Private Vulnerability Reporting for this repository.

If private reporting is unavailable, contact the maintainers at `isaac@dspy.ai`.

Include enough detail for the maintainers to reproduce, validate, and assess the report. As with ordinary bug reports, a minimal reproducible example or proof of concept will usually speed up triage.

At a minimum, please include:
- The affected version or commit SHA
- A clear description of the issue
- Steps to reproduce or a working proof of concept
- The expected security impact
- Any relevant logs, screenshots, or code snippets

Please redact secrets before sharing a report.

## Supported Versions

| Version | Supported |
| --- | --- |
| `main` | Yes |
| Latest stable release | Yes |
| Older releases | No |
| Pre-releases (`a`, `b`, `rc`) | No |

## Scope

Examples of security issues include:
- Arbitrary code execution
- Sandbox escape
- Path traversal or unsafe file access
- Server-side request forgery
- Deserialization vulnerabilities
- Secret or credential leakage caused by DSPy
- Supply-chain or release-process compromise in this repository

The following are not security issues by themselves:
- Model quality problems
- Hallucinations
- Prompt robustness issues
- Ordinary functional bugs without a clear security impact
- Vulnerabilities that exist only in third-party providers or downstream applications unless DSPy directly introduces the exposure

## Disclosure Process

We aim to acknowledge reports within 5 business days and coordinate a fix.

The resolution time will vary based on the reported severity and complexity. Timelines are best-effort and may vary based on report quality, severity, and volume.

Please avoid disclosing a vulnerability publicly until the maintainers have had a reasonable opportunity to investigate and remediate it.
