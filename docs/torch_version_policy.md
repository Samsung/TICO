# PyTorch Version Policy

TICO depends directly on `torch.export`, FX graphs, ATen operators, fake-tensor
behavior, and quantization internals. PyTorch compatibility therefore needs a
stricter policy than an ordinary leaf dependency.

The source of truth is
[`tico/utils/compat/torch_version_policy.py`](../tico/utils/compat/torch_version_policy.py).
The installer and GitHub Actions read this module instead of maintaining separate
hard-coded version lists.

## Package metadata and policy tiers

TICO deliberately separates package installation from qualified compatibility:

| Tier | Meaning | Current value |
|---|---|---|
| Package dependency | Allows installation into an existing Torch environment | Unbounded `torch` |
| Legacy installable | Accepted by `./ccex install`, but maintained on a best-effort basis | `2.5` through `2.9` |
| Supported stable | Qualified stable families used for release support | `2.10`, `2.11`, `2.12` |
| Default | Newest qualified stable family used by `./ccex install` | `2.12` |
| Candidate | Newly released stable family under qualification | `2.13` |
| Pinned nightly | Reproducible nightly channel for local debugging | `nightly` |
| Moving nightly | Latest compatible Torch/TorchVision nightly pair | `nightly-latest` |

`pyproject.toml` declares `"torch"` without a version specifier. Therefore an
already-installed older Torch release can satisfy `pip install tico`; package metadata
does not force an upgrade or reject it. This is an installation contract, not a claim
that every Torch release is qualified.

The source installer has explicit metadata for families 2.5 through 2.13. Families
2.5 through 2.9 are retained for existing users and reproducibility, but are not run in
the regular CI matrix. Their usability also depends on PyTorch publishing a wheel for
the requested Python, operating-system, and compute-platform combination.

## Nightly selectors

The two nightly selectors have deliberately different contracts:

```bash
# Install the exact Torch build from infra/dependency/torch_dev.txt. During
# test configuration, install the matching pinned TorchVision requirement.
./ccex install --torch_ver nightly
./ccex configure test --torch_ver nightly

# Resolve Torch and TorchVision together from the latest published nightly
# index, then preserve and validate that pair during test configuration.
./ccex install --torch_ver nightly-latest
./ccex configure test --torch_ver nightly-latest
```

`nightly-latest` installs `torch` and `torchvision` in the same pip resolver call.
This avoids selecting a moving Torch build and an independently resolved TorchVision
build. The composite GitHub Action delegates both selectors to `ccex`; it does not
maintain a separate nightly installation implementation.

Both selectors are re-resolved when explicitly requested. `nightly` restores the
repository pin, while `nightly-latest` checks the moving index for a newer compatible
pair. `configure test` validates that a pinned request matches the repository pin and
that a latest request already contains an importable nightly TorchVision build with a
matching CPU/CUDA wheel tag. Because the package dependency is unbounded, nightly
setup uses the normal `pip check` result and does not suppress a TICO/Torch metadata
conflict.

## CI tiers

| Trigger | PyTorch coverage | Test scope | Merge blocking |
|---|---|---|---|
| Every PR | Default family | Full unit-test suite | Yes |
| Every PR | Oldest supported family | Export and quantization smoke tests | Yes |
| Every PR | Candidate families, when configured | Export and quantization smoke tests | No |
| Daily schedule | `nightly-latest` | Export and quantization smoke tests | Separate workflow |
| Weekly schedule | All supported families | Full unit-test suite | Separate workflow |
| Weekly schedule | Candidate families and `nightly-latest` | Full unit-test suite | Separate workflow |
| Official release | All supported stable families | Full unit-test suite | Yes |

Legacy installable families are intentionally omitted from routine CI. A project that
still depends on one of those families should pin it explicitly and run its own
application-level verification.

The wheel is built once per workflow run and reused by all test jobs. PR jobs do not
upload one wheel per PyTorch family. Scheduled failures remain red so that maintainers
receive a useful upstream signal, but they do not participate in PR branch protection.

## Promoting a new stable family

When PyTorch `2.N` is released:

1. Add `2.N` to `QUALIFICATION_CANDIDATE_FAMILIES`.
2. Add its exact patch and published CUDA wheel variants to
   `LATEST_STABLE_VERSION` and `STABLE_CUDA_WHEELS`.
3. Keep it as an experimental candidate for 28 days after the candidate CI is
   merged while daily and weekly jobs collect compatibility results.
4. After qualification, append it to `SUPPORTED_STABLE_FAMILIES`, set it as
   `DEFAULT_FAMILY`, move the oldest supported family to
   `LEGACY_INSTALLABLE_FAMILIES`, and clear it from the candidate tuple.
5. Keep the package dependency unbounded unless TICO becomes fundamentally
   un-installable with a known range of Torch releases.

The policy unit test rejects overlapping or non-contiguous tiers, a default that is not
the newest supported family, incomplete installer metadata, a bounded package Torch
dependency, duplicated nightly selectors, and a scheduled matrix that uses the
reproducible pin instead of the moving nightly channel.

## Release branches

A `rel/*` branch should keep the default family it had when the branch was cut.
Patch releases of that family may be adopted, but the branch should not shift its
qualified three-family window unless a security or compatibility issue requires it. A
release branch may keep its own pinned `nightly` requirements for reproduction; moving
`nightly-latest` results are advisory only.
