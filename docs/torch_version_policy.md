# PyTorch Version Policy

TICO depends directly on `torch.export`, FX graphs, ATen operators, fake-tensor
behavior, and quantization internals. PyTorch compatibility therefore needs a
stricter policy than an ordinary leaf dependency.

The source of truth is
[`tico/utils/compat/torch_version_policy.py`](../tico/utils/compat/torch_version_policy.py).
The installer and GitHub Actions read this module instead of maintaining separate
hard-coded version lists.

## Support window

TICO keeps these tiers:

| Tier | Meaning | Current value |
|---|---|---|
| Default | Newest qualified stable family used by `./ccex install` | `2.12` |
| Supported stable | Latest three qualified stable families | `2.10`, `2.11`, `2.12` |
| Candidate | Newly released stable family under qualification | `2.13` |
| Pinned nightly | Reproducible nightly channel for local debugging | `nightly` |
| Moving nightly | Latest compatible Torch/TorchVision nightly pair | `nightly-latest` |

The package metadata allows the qualified families and the current candidate
(`torch>=2.10,<2.14`) while excluding the next unknown stable family. Exact default
pinning is left to `./ccex install`, because an exact `Requires-Dist` pin would
prevent users and CI from selecting another configured family.

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
matching CPU/CUDA wheel tag.

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

The wheel is built once per workflow run and reused by all test jobs. PR jobs do not
upload one wheel per PyTorch family. Scheduled failures remain red so that maintainers
receive a useful upstream signal, but they do not participate in PR branch protection.

Nightly builds are intentionally outside the stable package bound. Test setup ignores
only TICO's expected Torch metadata mismatch for a development build and still fails on
every other `pip check` conflict.

## Promoting a new stable family

When PyTorch `2.N` is released:

1. Add `2.N` to `QUALIFICATION_CANDIDATE_FAMILIES`.
2. Add its exact patch and published CUDA wheel variants to
   `LATEST_STABLE_VERSION` and `STABLE_CUDA_WHEELS`.
3. Keep it as an experimental candidate for 28 days after the candidate CI is
   merged while daily and weekly jobs collect compatibility results.
4. After qualification, append it to `SUPPORTED_STABLE_FAMILIES`, set it as
   `DEFAULT_FAMILY`, remove the oldest supported family, and clear it from the
   candidate tuple.
5. Update the bounded Torch requirement in `pyproject.toml` when either edge of
   the configured family window changes.

The policy unit test rejects a non-contiguous window, a default that is not the
newest supported family, missing version metadata, a stale package minimum, duplicated
nightly selectors, and a scheduled matrix that uses the reproducible pin instead of the
moving nightly channel.

## Release branches

A `rel/*` branch should keep the default family it had when the branch was cut.
Patch releases of that family may be adopted, but the branch should not shift its
three-family window unless a security or compatibility issue requires it. A release
branch may keep its own pinned `nightly` requirements for reproduction; moving
`nightly-latest` results are advisory only.
