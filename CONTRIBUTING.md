# Contributing to TurboQuant-MLX

Thanks for your interest! This document covers how to set up a dev
environment, the rules for submitting changes, and how to sign your
commits so the project's license chain stays clean.

## Code of Conduct

By participating in this project you agree to abide by the community
[Code of Conduct](CODE_OF_CONDUCT.md) — be respectful, assume good faith,
and keep discussion focused on the work.

## Development setup

```bash
git clone https://github.com/manjunathshiva/turboquant-mlx.git
cd turboquant-mlx
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,eval]"
```

You'll need macOS on Apple Silicon, Xcode Command Line Tools, and
CMake 3.27+ to build the Metal kernels.

## Running tests

```bash
pytest tests/
```

48 tests should pass on a clean checkout. The KV cache tests require
`mlx-lm>=0.31.3`.

## How to propose a change

1. Open a GitHub issue first to discuss substantial changes (new kernels,
   new architecture support, API breaks). Small fixes can go straight to a PR.
2. Fork, branch, implement, add tests, run the suite, open a PR.
3. Keep PRs focused — one logical change per PR makes review tractable.
4. Match existing code style (PEP 8, type hints on public APIs, docstrings
   on exported functions).

## Sign-off (Developer Certificate of Origin)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) instead of a contributor-license agreement. By signing off your
commits, you certify that you wrote the code (or have the right to
contribute it) and agree to release it under the project's license
(Apache-2.0 for v0.3.0 and later).

To sign off, add a `Signed-off-by` trailer to every commit:

```bash
git commit -s -m "your commit message"
```

`-s` is a shortcut that automatically appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Configure git once with the name and email you want to appear in the trailer:

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

PRs without sign-off will be asked to add it before merge. There is no
separate CLA to sign and no rights assignment — you keep copyright over
your contribution; the DCO simply confirms you have the right to
contribute it under Apache-2.0.

The full DCO text is at https://developercertificate.org/ (it's short —
one screen).

## Reporting security issues

Do **not** open a public issue for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for the disclosure policy.

## License

By contributing to this project, you agree that your contributions will
be licensed under the [Apache License 2.0](LICENSE) for v0.3.0 and later
releases.
