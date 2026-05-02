# Security Policy

## Supported Versions

`turboquant-mlx-full` follows semantic versioning. Security fixes ship in the
latest minor release. The previous minor receives a fix only if the issue is
actively exploited and there is no reasonable upgrade path for users.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x: (please upgrade)|

## Reporting a Vulnerability

If you believe you have found a security issue in `turboquant-mlx-full`,
please **do not** open a public GitHub issue. Instead:

1. Email **manjunath.shiva@gmail.com** with the subject line
   `[turboquant-mlx security] <short summary>`.
2. Include enough detail to reproduce: the affected version, a minimal code
   sample, and the observed vs expected behavior. If you have a proof-of-concept,
   include or link to it.
3. You should expect an acknowledgement within **5 business days**. The
   maintainer will share an estimated remediation timeline once the report
   is triaged.

While the report is being investigated, please refrain from publicly disclosing
the issue (issue tracker, social media, blog posts, conference talks). Once a
fix has shipped and a reasonable window has passed for users to upgrade, you
are encouraged to publish your write-up — credit will be given in the release
notes unless you prefer to remain anonymous.

## Scope

This policy covers issues in:

- The Python source tree (`turboquant_mlx/` and subpackages) shipped on PyPI as
  `turboquant-mlx-full`.
- The C++/Metal kernel sources under `csrc/`.
- Default configurations (`pyproject.toml`, build scripts) that affect what gets
  installed on a user's machine.

Out of scope:

- Vulnerabilities in upstream dependencies (`mlx`, `mlx-lm`, etc.) — please
  report those to the relevant project. We will, however, ship a release that
  pins or works around a confirmed upstream issue if it materially affects
  TurboQuant users.
- Issues that require a malicious actor to already have local code-execution
  rights on the user's machine.
- Quality-of-output regressions in quantized models. Those are bugs and should
  be filed as regular GitHub issues.

## Scoped tokens

This project's PyPI publishing uses scoped API tokens (per project, not account
wide). The maintainer rotates the publishing token after each release.
