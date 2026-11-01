# Tiny Gemma4 attention R0K1 vs R1K1 (own-cache rollout)

- R0K1: tico `1f9fac105602` dirty_diff_sha256 `e3b0c44298fc` attention_src `e94dbc5dbbdf` policy=None
- R1K1: tico `1f9fac105602` dirty_diff_sha256 `aa751e456595` attention_src `759d30d147fa` policy={'k': True, 'q': True}

## profile `affine`

| seed | metric | R0K1 | R1K1 | R1K1-R0K1 |
|---|---|---:|---:|---:|
| 123 | prefill_mae | 0.02317641 | 0.02317641 | +0.00000000 |
| 123 | mean_decode_mae_1_128 | 0.03610782 | 0.03589801 | -0.00020981 |
| 123 | mean_decode_mae_1_32 | 0.03203308 | 0.03128153 | -0.00075155 |
| 123 | step32_mae | 0.00896046 | 0.00896046 | +0.00000000 |
| 123 | step128_mae | 0.08681359 | 0.08617357 | -0.00064002 |
| 123 | step128_rmse | 0.10810972 | 0.10813072 | +0.00002100 |
| 123 | step128_max_abs | 0.26967886 | 0.26967886 | +0.00000000 |
| 123 | step128_normalized_l2 | 0.26994184 | 0.26999427 | +0.00005243 |
| 123 | mean_decode_replay_mae | 0.03141212 | 0.03141212 | +0.00000000 |
| 123 | mean_kv_delta_mae_K | 0.01603518 | 0.01595984 | -0.00007534 |
| 123 | mean_kv_delta_mae_V | 0.01281118 | 0.01281118 | +0.00000000 |
| 123 | circle_recurrent_mean_fp_vs_circle_mae | 0.03610782 | 0.03589801 | -0.00020981 |
| 123 | circle_recurrent_max_same_cache_eager_vs_circle | 0.00000000 | 0.00000000 | +0.00000000 |
| 123 | paired per-step MAE diff (mean/min/max) | | | -2.098e-04 / -2.699e-02 / +3.808e-03 (R1 better 6/128 steps) |
| 123 | Quantize total (prefill_O1 / decode_O1) | 4 / 4 | 0 / 0 | |
| 123 | common qparam differences (non-RoPE sites) | | | 0 |
| 456 | prefill_mae | 0.01506523 | 0.01506523 | +0.00000000 |
| 456 | mean_decode_mae_1_128 | 0.03357015 | 0.03341939 | -0.00015076 |
| 456 | mean_decode_mae_1_32 | 0.03958820 | 0.03939021 | -0.00019799 |
| 456 | step32_mae | 0.12990171 | 0.12990171 | +0.00000000 |
| 456 | step128_mae | 0.03605375 | 0.03605375 | +0.00000000 |
| 456 | step128_rmse | 0.04549422 | 0.04549422 | +0.00000000 |
| 456 | step128_max_abs | 0.12151971 | 0.12151971 | +0.00000000 |
| 456 | step128_normalized_l2 | 0.13889177 | 0.13889177 | +0.00000000 |
| 456 | mean_decode_replay_mae | 0.02678313 | 0.02689667 | +0.00011353 |
| 456 | mean_kv_delta_mae_K | 0.01651538 | 0.01645823 | -0.00005715 |
| 456 | mean_kv_delta_mae_V | 0.01232852 | 0.01232852 | +0.00000000 |
| 456 | circle_recurrent_mean_fp_vs_circle_mae | 0.03354911 | 0.03339835 | -0.00015076 |
| 456 | circle_recurrent_max_same_cache_eager_vs_circle | 0.00000000 | 0.00000000 | +0.00000000 |
| 456 | paired per-step MAE diff (mean/min/max) | | | -1.508e-04 / -1.589e-02 / +7.717e-03 (R1 better 5/128 steps) |
| 456 | Quantize total (prefill_O1 / decode_O1) | 6 / 6 | 0 / 0 | |
| 456 | common qparam differences (non-RoPE sites) | | | 0 |
| 789 | prefill_mae | 0.01372997 | 0.01372997 | +0.00000000 |
| 789 | mean_decode_mae_1_128 | 0.03773284 | 0.03770852 | -0.00002432 |
| 789 | mean_decode_mae_1_32 | 0.03458985 | 0.03455548 | -0.00003437 |
| 789 | step32_mae | 0.03881814 | 0.03881814 | +0.00000000 |
| 789 | step128_mae | 0.02453921 | 0.02453921 | +0.00000000 |
| 789 | step128_rmse | 0.03042815 | 0.03042815 | +0.00000000 |
| 789 | step128_max_abs | 0.07395689 | 0.07395689 | +0.00000000 |
| 789 | step128_normalized_l2 | 0.09054151 | 0.09054151 | +0.00000000 |
| 789 | mean_decode_replay_mae | 0.03308345 | 0.03307044 | -0.00001301 |
| 789 | mean_kv_delta_mae_K | 0.01623527 | 0.01622273 | -0.00001254 |
| 789 | mean_kv_delta_mae_V | 0.01279585 | 0.01279585 | +0.00000000 |
| 789 | circle_recurrent_mean_fp_vs_circle_mae | 0.03773284 | 0.03770852 | -0.00002432 |
| 789 | circle_recurrent_max_same_cache_eager_vs_circle | 0.00000000 | 0.00000000 | +0.00000000 |
| 789 | paired per-step MAE diff (mean/min/max) | | | -2.432e-05 / -2.014e-03 / +0.000e+00 (R1 better 2/128 steps) |
| 789 | Quantize total (prefill_O1 / decode_O1) | 4 / 4 | 0 / 0 | |
| 789 | common qparam differences (non-RoPE sites) | | | 0 |
| mean | prefill_mae | 0.01732387 | 0.01732387 | +0.00000000 |
| mean | mean_decode_mae_1_128 | 0.03580360 | 0.03567530 | -0.00012830 |
| mean | mean_decode_mae_1_32 | 0.03540371 | 0.03507574 | -0.00032797 |
| mean | step32_mae | 0.05922677 | 0.05922677 | +0.00000000 |
| mean | step128_mae | 0.04913552 | 0.04892218 | -0.00021334 |
| mean | step128_rmse | 0.06134403 | 0.06135103 | +0.00000700 |
| mean | step128_max_abs | 0.15505182 | 0.15505182 | +0.00000000 |
| mean | step128_normalized_l2 | 0.16645837 | 0.16647585 | +0.00001748 |
| mean | mean_decode_replay_mae | 0.03042623 | 0.03045974 | +0.00003351 |
| mean | mean_kv_delta_mae_K | 0.01626194 | 0.01621360 | -0.00004834 |
| mean | mean_kv_delta_mae_V | 0.01264519 | 0.01264519 | +0.00000000 |
| mean | circle_recurrent_mean_fp_vs_circle_mae | 0.03579659 | 0.03566829 | -0.00012830 |
| mean | circle_recurrent_max_same_cache_eager_vs_circle | 0.00000000 | 0.00000000 | +0.00000000 |

## profile `u8s16c`

| seed | metric | R0K1 | R1K1 | R1K1-R0K1 |
|---|---|---:|---:|---:|
| 123 | prefill_mae | 0.00396759 | 0.00396759 | +0.00000000 |
| 123 | mean_decode_mae_1_128 | 0.00960156 | 0.00935495 | -0.00024660 |
| 123 | mean_decode_mae_1_32 | 0.00713948 | 0.00703809 | -0.00010139 |
| 123 | step32_mae | 0.00433874 | 0.00433874 | +0.00000000 |
| 123 | step128_mae | 0.04841694 | 0.04841694 | +0.00000000 |
| 123 | step128_rmse | 0.05954163 | 0.05954163 | +0.00000000 |
| 123 | step128_max_abs | 0.14091641 | 0.14091641 | +0.00000000 |
| 123 | step128_normalized_l2 | 0.14867097 | 0.14867097 | +0.00000000 |
| 123 | mean_decode_replay_mae | 0.00908558 | 0.00908052 | -0.00000506 |
| 123 | mean_kv_delta_mae_K | 0.00712716 | 0.00706199 | -0.00006517 |
| 123 | mean_kv_delta_mae_V | 0.00712394 | 0.00712394 | +0.00000000 |
| 123 | circle_recurrent_mean_fp_vs_circle_mae | 0.00960231 | 0.00935567 | -0.00024664 |
| 123 | circle_recurrent_max_same_cache_eager_vs_circle | 0.00022089 | 0.00033128 | +0.00011039 |
| 123 | paired per-step MAE diff (mean/min/max) | | | -2.466e-04 / -1.876e-02 / +2.698e-03 (R1 better 50/128 steps) |
| 123 | Quantize total (prefill_O1 / decode_O1) | 6 / 8 | 2 / 4 | |
| 123 | common qparam differences (non-RoPE sites) | | | 0 |
| 456 | prefill_mae | 0.00293242 | 0.00293242 | +0.00000000 |
| 456 | mean_decode_mae_1_128 | 0.00983283 | 0.00979413 | -0.00003870 |
| 456 | mean_decode_mae_1_32 | 0.00999180 | 0.00996278 | -0.00002902 |
| 456 | step32_mae | 0.05314872 | 0.05314872 | +0.00000000 |
| 456 | step128_mae | 0.02275906 | 0.02275906 | +0.00000000 |
| 456 | step128_rmse | 0.02783394 | 0.02783394 | +0.00000000 |
| 456 | step128_max_abs | 0.06614762 | 0.06614762 | +0.00000000 |
| 456 | step128_normalized_l2 | 0.08497575 | 0.08497575 | +0.00000000 |
| 456 | mean_decode_replay_mae | 0.00816086 | 0.00813764 | -0.00002321 |
| 456 | mean_kv_delta_mae_K | 0.00730962 | 0.00730877 | -0.00000086 |
| 456 | mean_kv_delta_mae_V | 0.00688474 | 0.00688474 | +0.00000000 |
| 456 | circle_recurrent_mean_fp_vs_circle_mae | 0.00983519 | 0.00979648 | -0.00003870 |
| 456 | circle_recurrent_max_same_cache_eager_vs_circle | 0.00102890 | 0.00102890 | +0.00000000 |
| 456 | paired per-step MAE diff (mean/min/max) | | | -3.870e-05 / -2.005e-03 / +6.770e-06 (R1 better 4/128 steps) |
| 456 | Quantize total (prefill_O1 / decode_O1) | 6 / 8 | 2 / 4 | |
| 456 | common qparam differences (non-RoPE sites) | | | 0 |
| 789 | prefill_mae | 0.00320930 | 0.00320930 | +0.00000000 |
| 789 | mean_decode_mae_1_128 | 0.01695049 | 0.01696800 | +0.00001752 |
| 789 | mean_decode_mae_1_32 | 0.01270071 | 0.01274923 | +0.00004853 |
| 789 | step32_mae | 0.01663298 | 0.01663298 | +0.00000000 |
| 789 | step128_mae | 0.01294808 | 0.01294808 | +0.00000000 |
| 789 | step128_rmse | 0.01605435 | 0.01605435 | +0.00000000 |
| 789 | step128_max_abs | 0.04450323 | 0.04450323 | +0.00000000 |
| 789 | step128_normalized_l2 | 0.04777108 | 0.04777108 | +0.00000000 |
| 789 | mean_decode_replay_mae | 0.01535364 | 0.01537714 | +0.00002350 |
| 789 | mean_kv_delta_mae_K | 0.00872424 | 0.00872424 | +0.00000000 |
| 789 | mean_kv_delta_mae_V | 0.00760211 | 0.00760211 | +0.00000000 |
| 789 | circle_recurrent_mean_fp_vs_circle_mae | 0.01694963 | 0.01696716 | +0.00001753 |
| 789 | circle_recurrent_max_same_cache_eager_vs_circle | 0.00080866 | 0.00080866 | +0.00000000 |
| 789 | paired per-step MAE diff (mean/min/max) | | | +1.752e-05 / -1.820e-04 / +9.977e-04 (R1 better 2/128 steps) |
| 789 | Quantize total (prefill_O1 / decode_O1) | 6 / 8 | 2 / 4 | |
| 789 | common qparam differences (non-RoPE sites) | | | 0 |
| mean | prefill_mae | 0.00336977 | 0.00336977 | +0.00000000 |
| mean | mean_decode_mae_1_128 | 0.01212829 | 0.01203903 | -0.00008926 |
| mean | mean_decode_mae_1_32 | 0.00994400 | 0.00991670 | -0.00002730 |
| mean | step32_mae | 0.02470681 | 0.02470681 | +0.00000000 |
| mean | step128_mae | 0.02804136 | 0.02804136 | +0.00000000 |
| mean | step128_rmse | 0.03447664 | 0.03447664 | +0.00000000 |
| mean | step128_max_abs | 0.08385575 | 0.08385575 | +0.00000000 |
| mean | step128_normalized_l2 | 0.09380593 | 0.09380593 | +0.00000000 |
| mean | mean_decode_replay_mae | 0.01086669 | 0.01086510 | -0.00000159 |
| mean | mean_kv_delta_mae_K | 0.00772034 | 0.00769833 | -0.00002201 |
| mean | mean_kv_delta_mae_V | 0.00720360 | 0.00720360 | +0.00000000 |
| mean | circle_recurrent_mean_fp_vs_circle_mae | 0.01212904 | 0.01203977 | -0.00008927 |
| mean | circle_recurrent_max_same_cache_eager_vs_circle | 0.00068615 | 0.00072295 | +0.00003680 |

