# Releasing to PyPI

The package builds cleanly and the release is automated by
[`.github/workflows/release.yml`](.github/workflows/release.yml), which publishes to PyPI
via **Trusted Publishing (OIDC)** — no API token is stored anywhere.

Publishing needs a **one-time setup** (only the project owner can do this), then each
release is just a tag.

## One-time setup (owner)

1. **Create the PyPI project via Trusted Publishing** (no token needed):
   - Log in to <https://pypi.org> → *Account* → *Publishing* → *Add a pending publisher*.
   - Fill in:
     - **PyPI project name:** `ds-msp`
     - **Owner:** `Munna-Manoj`
     - **Repository name:** `DS-MSP`
     - **Workflow name:** `release.yml`
     - **Environment name:** `pypi`
2. **Create the GitHub environment** named `pypi`:
   - GitHub repo → *Settings* → *Environments* → *New environment* → `pypi`.
   *(Optional but recommended: add required reviewers so a human approves each publish.)*

> Prefer an API token instead of OIDC? Add a `PYPI_API_TOKEN` repo secret and pass it to
> the publish step with `with: { password: ${{ secrets.PYPI_API_TOKEN }} }`. Trusted
> Publishing is recommended because there is no long-lived secret to leak.

## Cutting a release

1. Bump `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
2. Verify locally:
   ```bash
   python -m build          # or: uv build
   twine check dist/*
   ```
3. Commit, then tag and push:
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```
4. The **Release** workflow builds, runs `twine check`, and publishes. Confirm at
   <https://pypi.org/project/ds-msp/>.
5. After the first successful publish, add the install line to the README:
   `pip install ds-msp`.

## Local build sanity check

```bash
uv build && uv pip install --force-reinstall dist/ds_msp-*.whl
python -c "import ds_msp; print('ok')"
```
