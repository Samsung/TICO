# Quantization analysis

`TICO.quantization.analysis` contains model-independent tools for measuring
post-training quantization error. It operates on prepared and calibrated WrapQ
models and does not own model loading, calibration datasets, task metrics, or
backend export.

## Standard ablation profiles

`QuantizationAblation` uses observer-level runtime switches to evaluate:

| Profile | Enabled sites |
|---|---|
| A — output-only | explicitly selected final-output domains |
| B — weight-only | parameter observers |
| C — activation-only | internal non-parameter sites, excluding selected outputs |
| D — full | all included sites |
| E — internal-full | all included sites except selected final-output domains |

A model adapter supplies `QuantizationBoundaries.outputs`; the other default
profiles are derived from observer roles. More specialized models may also
supply explicit parameter, activation, or included selectors. Selectors may
match full observer paths, owner-module paths or types, observer names, and
logical quantization roles.

Comparing E with D isolates the effect of the selected output domains after
parameters and internal activations are already quantized.

```python
from tico.quantization.analysis import (
    QuantizationAblation,
    QuantizationBoundaries,
    SiteSelector,
    make_output_adapter,
)

boundaries = QuantizationBoundaries(
    outputs=(
        SiteSelector.module_paths("model.output_head")
        & SiteSelector.observer_names("act_out")
    )
)

report = QuantizationAblation(
    float_model,
    calibrated_quantized_model,
    boundaries=boundaries,
    output_adapter=make_output_adapter(("logits",)),
).run(evaluation_samples)
```

## Output clipping

`collect_output_calibration_data`, `build_clipping_candidates`, and
`evaluate_clipping_candidates` compare output-only MinMax, percentile, and L1
grid-search clipping without quantizing internal model operations.

The calibration and evaluation datasets may be the same for numerical-floor
analysis. Selecting a production policy or reporting final accuracy requires an
independent evaluation set.

## Sensitivity

`QuantizationSensitivity` evaluates named `QuantizationGroup` selectors in two
modes:

- `LEAVE_ONE_FLOAT`: start from full fake quantization and disable groups;
- `ENABLE_ONE`: start from floating-point behavior and enable groups.

`run()` measures every group independently. `run_cumulative()` applies groups
in the supplied order and reports both incremental and total sensitivity.
A caller can feed the initial `run()` ranking into `run_cumulative()` for a
fixed-ranked path. `run_greedy()` re-evaluates every remaining candidate after
each selected group, so it captures interactions that independent
leave-one-float scores cannot be added to estimate.

Groups may contain one observer or a complete precision domain such as a
producer output, consumer input, and shared Add/Concat boundary. Both cumulative
and greedy paths reject groups that add no mutable quantization site relative to
the selected baseline.

Pass `baseline_selector` to analyze a custom profile rather than the default
all-enabled or all-disabled baseline. For example, an E:internal-full selector
keeps final-output domains floating point while leave-one-float analysis ranks
internal groups:

```python
baseline_selector = boundaries.selector_for(
    QuantizationProfile.INTERNAL_FULL
)
baseline, results = QuantizationSensitivity(
    float_model,
    calibrated_quantized_model,
).run(
    evaluation_samples,
    groups,
    mode=SensitivityMode.LEAVE_ONE_FLOAT,
    score_output="logits",
    baseline_selector=baseline_selector,
)
```

A group is matched only against sites that can change relative to the selected
baseline. This prevents a leave-one-float group from silently targeting sites
that are already floating point.

## Runtime control

Observer fake-quantization switches are independent from calibration
collection. `tico.quantization.wrapq.control.FakeQuantState` snapshots and
restores them so analysis does not permanently modify a candidate model.
