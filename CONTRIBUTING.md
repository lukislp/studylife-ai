# Contributing to StudyLife AI

Thanks for taking the time. StudyLife AI is a single-maintainer project, so the process is
deliberately small - but it is the same for every change, including the maintainer's own.

## How changes get in

1. Open an issue first for anything bigger than a typo or an obvious bug fix, so the direction can
   be agreed before you spend time on it. Use the templates under `.github/ISSUE_TEMPLATE/`.
2. Fork the repository (or branch, if you have write access) and make your change on a branch.
3. Open a pull request against `main`. The pull-request template asks for what changed and why.
4. `main` is protected: a PR merges only after the test stage of
   [`.github/workflows/ci.yml`](.github/workflows/ci.yml) is green and the branch is up to date
   with `main` (enable auto-merge and it lands on its own once that is the case). Nobody pushes to
   `main` directly, not even the maintainer.

## What a pull request needs

- **Conventional Commits.** The version and the changelog are generated from the commit messages
  (`feat:` = minor release, `fix:` = patch release, `build:`/`ci:`/`docs:`/`test:` = no release).
  Squash-merge keeps the PR title as the commit message, so give the PR a Conventional Commit
  title.
- **Green required checks.** `lint`, `test` and `review / dependency-review` are required; a red
  one blocks the merge.
- **Tests for new functionality.** New features and bug fixes come with tests in `tests/`
  (`tests/agent/` for the graph and its tools, `tests/contract/` for the API contract). A PR that
  adds behaviour without a test is asked to add one.
- **Lint, formatting and types.** The `lint` job runs three separate gates - `ruff check .`,
  `ruff format --check .` and `mypy src`. mypy is in **strict** mode here, so new code needs real
  annotations rather than `Any`. Run all three before pushing; `ruff format --check` in particular
  is easy to miss, since `ruff check` passing does not mean the formatting is clean.
- **API contract.** `tests/contract/test_openapi_contract.py` runs as part of the ordinary `pytest`
  call and checks this service against the main `studylife` repository's committed
  `docs/api/openapi.json`. It **fails** rather than skips if the spec is unreachable or has
  drifted, so a contract change has to land on both sides.
- **Lockfile.** CI installs with `uv sync --frozen`, so a dependency change means committing the
  updated `uv.lock` alongside `pyproject.toml`.

## Running things locally

Python 3.12 or newer, with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn studylife_ai.main:app --reload
```

The gates CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

`.pre-commit-config.yaml` wires ruff, `ruff-format`, the usual whitespace hooks and `mypy src` into
a commit hook - `uv run pre-commit install` once and the first three gates run on every commit.

The fuzz target is Linux-only (`atheris` publishes no Windows wheel):

```bash
uv sync --frozen --group fuzz
uv run python fuzz/fuzz_prompt_and_schemas.py -max_total_time=30 -rss_limit_mb=1024
```

## The evaluation pipeline

`eval/` holds the RAGAS dataset and fixtures. The eval runs from
[`.github/workflows/eval.yml`](.github/workflows/eval.yml) on pushes to `main` only, not on pull
requests, and it gates on the run not raising - there are no score thresholds yet. Locally:

```bash
uv run python -m studylife_ai.eval.seed_fixture   # seed the fixture corpus
uv run python -m studylife_ai.eval               # needs EVAL_JUDGE_MODEL
```

If you change retrieval, chunking or the prompt, say in the PR whether you ran the eval and what
moved.

## Security issues

Please do not open a public issue for a vulnerability - use the private reporting path described
in [SECURITY.md](SECURITY.md). The [Code of Conduct](CODE_OF_CONDUCT.md) applies to every
interaction in this repository.
