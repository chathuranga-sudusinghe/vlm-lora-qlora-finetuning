# Local data

Keep raw images, prepared examples, split manifests, and dataset downloads local.
Everything in this folder except this README is ignored by the issue #5 ignore
rules. No standalone `sample.jpg` is required or needs to be recreated.

Use the Hugging Face cache or, when preparation is implemented, `data/raw/` and
`data/processed/`. Dataset selection and integrity rules live in
[docs/dataset.md](../docs/dataset.md). Raw files being present does not establish
validated rows/images, verified download revisions, checksums, or prepared splits.
Preserve snapshot identifiers, audit decisions, vocabulary, and split membership
locally when preparation is implemented; generated data does not belong in Git.

Do not add downloaded images as test fixtures; generate small synthetic images
when a test needs one. Small intentional summary evidence can be reviewed separately.
