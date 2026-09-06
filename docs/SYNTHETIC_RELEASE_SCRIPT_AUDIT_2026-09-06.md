# Synthetic Web release-boundary review

This review did not open the new independent evaluation examples and did not
modify frozen classifier, inference, splitter, grounding, or game-policy files.

## Corrections

`scripts/promote_synthetic_web.py` now refuses absent/empty/false or non-boolean
gates, missing classes, nonfinite metrics, inconsistent counts, failed mixed
components, changed/missing model and inference hashes, and incomplete stateful
acceptance. `--stateful-report` is required: the same 20 uncertain IDs must appear
in both prescribed game contexts, with every update rejected. The report must
bind to the same dataset/predictions and current grounding, game, Pragmatic,
VADER implementation, and VADER-table hashes. All checks finish before release
assets are written; publication uses atomic file replacement.

`scripts/prepare_sites_synthetic_source.ps1` now exactly synchronizes the approved
source files into the isolated clone. Removed routes or public assets no longer
remain from an older checkout. It refuses environment/private/database files and
linked source inputs before mutation. Stale files are removed individually after
checking the absolute target and its ancestors remain inside the intended
workspace. It never traverses or deletes preserved dependency junctions, `.git`,
or generated build directories.

`scripts/package_synthetic_site.py --build` requires a clean committed source,
inventories all non-generated source files, clears the explicitly bounded prior
`dist`, runs the build, and checks that source and HEAD stayed unchanged. It then
issues a receipt binding HEAD, source inventory, and compiled-asset inventory.
Packaging without `--build` requires a matching existing receipt. A new clean
commit cannot relabel an older ignored `dist` as its own build. Archives contain
only root `.openai` metadata and `dist/client`, `dist/server` assets; environment/database/linked members
and known secret values are rejected, and archive bytes are checked against the
receipt inventory.

## Verification

`testing/test_synthetic_release_scripts.py`: initially 10 tests passed, including multiple
failure subcases. Tests use isolated fake Git repositories, an actual dummy Node
build, synthetic metadata and synthetic secret strings. They cover changed source
commits, changed compiled files, ignored environment files, secret/database
exclusion, precise preparation, failed/malformed acceptance, missing source hashes,
Windows directory-junction exclusion, and incomplete or unsafe stateful reports. Failed promotion preserves the old
manifest and writes no new model assets.

`web/app/model-check/page.tsx` was independently read with its loader and layout.
It performs static model downloads and local classification, without importing
the research client, creating a session, enqueueing events, or submitting entered
text. Its statement that this page does not submit research data matches the code.

Deployment status and remote source revision are established separately by the
release workflow after the checked archive and successful source push exist.

## Sites archive layout correction

The first save attempt rejected the archive because its root `server/index.js`
was not a supported entrypoint. Packaging now maps build `.openai/*` to archive
`.openai/*`, build `client/*` to `dist/client/*`, and build `server/*` to
`dist/server/*`. Sites can discover `dist/server/index.js`, while the archived
`dist/server/wrangler.json` keeps `main=index.js` and `assets.directory=../client`
resolving to that Worker and `dist/client`. The build receipt retains its original
source and dist inventory; only the tar layout changes. Each sidecar inventory
row records both the archive `path` and original `build_path`, plus SHA and size.
Archive verification checks the complete mapped member set, regular-file type,
size and per-file hash after the existing secret and receipt validation.

The eight focused `PackagingBoundaryTests` passed after this correction. Two
added fake-repository tests cover supported entrypoint discovery, root hosting
metadata, Wrangler's relative assets, unchanged receipt bytes when packaging
without a build, and refusal of local `dist/server/.wrangler` database state.
Existing archives are not overwritten. The release checkout was observed to
contain local `.wrangler` state after preview; the release owner will use the
existing bounded `--build` cleanup and issue a fresh build receipt, not silently
exclude unverified generated members.
