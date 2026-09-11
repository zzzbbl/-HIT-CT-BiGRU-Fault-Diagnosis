# Release Checklist

Before publishing a release or pushing to a public GitHub repository, verify:

- no raw datasets are tracked
- no processed tensors are tracked
- no model checkpoints are tracked
- no generated figures or experiment outputs are tracked
- no passwords, API keys, access tokens, or private credentials are present
- no local absolute paths are present
- no personal contact information is present unless intentionally included for
  scholarly citation or maintainer identity
- no Python caches, notebook checkpoints, editor files, or operating-system
  temporary files are present
- `python -m compileall src` passes
- `git status --short --ignored` only shows expected tracked and ignored files

