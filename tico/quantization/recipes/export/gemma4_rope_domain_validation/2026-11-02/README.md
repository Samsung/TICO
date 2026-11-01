# Gemma4 RoPE domain inheritance (R1K1) production 검증 — 2026-11-02

기준은 Samsung/TICO PR #1101 head `1f9fac105602767678bf733f7a0ca82aa97ae5d0`이다.
작업 시점에 #1101은 upstream `main`에 squash 머지되어 있었고(`1d8f2983`), 두 tree는
동일했다(`git diff --stat 1f9fac10 upstream/main` 비어 있음). 구현은 `1f9fac10` 위의
별도 worktree/브랜치 `feat/gemma4-rope-domain-unification`에서 했고, 구현 커밋은
`6ccf489468477041c2d74f7858da604065e79e3a`이다. 사용자 변경(원래 checkout의 untracked
파일)은 건드리지 않았다.

R0K1 control은 **수정하지 않은 `1f9fac10`**을 별도 worktree에 checkout하여 별도 process로
실행했다. 모든 run은 자신이 import한 `tico`의 git HEAD, dirty diff hash, attention source
hash를 기록하고, 실제 실행된 정책(`_rope_domains` 유무, half/cat observer의 frozen/active
여부)을 label과 무관하게 assertion으로 확인했다.

## 구현한 정책

`QuantGemma4TextAttention`은 생성 시점에 branch별(Q, K)로 다음 조건을 모두 만족하면
"norm output domain 상속"으로 고정한다(`_rope_domains`).

- `rope="pre_negated_sin"` (npu_export profile의 기본값)
- 해당 branch의 norm(`q_norm`/`k_norm`) 출력 observer가 per-tensor affine
- `q_x1/q_x2/q_cat`(`k_x1/k_x2/k_cat`)가 per-tensor affine이고 **명시 override가 없음**
- K는 producer layer에만 해당(shared-KV consumer는 `k_norm`이 없고 Q만 판정)

상속 branch에서 `_rot()`은 CALIB에서 아무것도 수집하지 않고(norm observer는 원래 위치에서
forward당 1회만 수집), QUANT에서는 이미 frozen인 norm grid의 `fake_quant`를 x1/x2/cat에
재선언만 한다(derive/수정 없음). 일반 exporter가 중복 marker를 fold하므로 serialized
norm output → halves → reordered concat은 하나의 domain이 되고 그 사이 Quantize는 없다.
cos/sin 입력, `*_cos/*_sin/*_rot`, logits/softmax/output, weight, S/C cache grid,
mask/position/RoPE table 준비 규칙은 그대로다. `hf` convention, per-channel, MX, identity,
명시 override 사이트는 기존 독립 observer 경로를 유지한다("기존 경로 유지" 정책). 상속
branch의 norm output fake quant를 끄면 freeze 시 `ValueError`로 거부한다.

정책은 attention extra state `gemma_rope`에 기록된다. 다른 정책으로 만든 state dict나
whole-module checkpoint는 `strict=False`에서도 `RuntimeError`로 거부한다. 이 기록이 없는
과거 checkpoint는 wrapper가 여전히 독립 grid로 해석될 때만(예: `rope="hf"`, 명시 사이트
override) 로드되고, 상속이 적용되는 구성에서는 fresh prepare/calibrate를 요구한다. 자동
변환은 없다. 기존 K1(`F_S(r)` 저장, `F_C(r)` current, `F_C(p_S)` past, C/C→C concat,
S delta append)은 수정하지 않았고 보존 테스트를 추가했다.

구현 중 발견한 함정: torch.export 추적 중 `self.obs_q_x1` 같은 submodule 접근은 매번 새
attribute proxy를 돌려주므로 observer identity(`obs is self.obs_q_x1`)로 branch를 판별하면
export에서 fallback 경로로 빠진다. production은 `_apply_rope/_rot`에 branch 키를 명시적으로
전달한다. 기존 실험 subclass(identity 판별)는 이 환경(torch 2.13)에서 export가 되지 않았다.

함께 추가한 것: cached-decode NLL/PPL 평가(`recipes/evaluation/cached_nll.py`,
Gemma4 `cached_nll` task). 실제 prefill 1회 + target 토큰당 cached decode 1 step으로
held-out chunk를 teacher forcing하여 past-KV 읽기 경로를 실제로 측정한다.

## 로컬 검증 명령과 결과

환경: Python 3.10.4, torch 2.13.0+cu126, transformers 5.18.0, CPU. 명령/종료 코드는
[validation.json](validation.json)에, artifact hash는
[artifact-manifest.json](artifact-manifest.json)에 있다.

| 검증 | 결과 |
|---|---|
| 기존 Gemma4 attention/cache/export/lifecycle 테스트 9개 모듈 (`python -m unittest …`) | 105 tests OK |
| 신규 `test_rope_domain_inheritance.py` 16 + `test_cached_nll.py` 5 | 21 tests OK |
| `./ccex test -k gemma4` | 742 tests OK (skipped 70, 기존 skip) |
| `./ccex test` (전체 non-model suite) | 4818 tests OK (skipped 157, 기존 skip) |
| `lintrunner` (변경/신규 파일) | No lint issues |

신규 테스트가 다루는 것: NO_QUANT/CALIB 출력과 norm 통계가 독립 경로와 동일, canonical
norm observer collect 횟수(MinMax/EMA) forward당 1회 유지, 상속 사이트 grid 없음/독립
경로 grid 있음, Q/K 독립 판정(`k_cat` override, dtype override, per-channel/MX/float norm),
shared-KV consumer의 Q-only 상속과 `_freeze_dependencies`, lifecycle 전반 inventory 불변,
bypass norm output 거부, state_dict/whole-module/deepcopy round trip, 다른 정책 checkpoint
거부(strict True/False, legacy record), export 성공/실패 후 state fingerprint 불변,
K1 cache 계약, UINT8/mixed profile의 serialized RoPE domain·Quantize 수·S→C 변환·FAKE_QUANTIZE parity.

## Tiny R0K1 대 R1K1 (fixed seed-123 weights, 3 data seed, own-cache 128 step)

실험 fixture와 동일: smoke tiny attention(hidden 64, 2 head, head_dim 32), full attention,
capacity 136, calibration 3 stream × (8 prefill + 4 cached decode) = 36 token, evaluation
별도 stream 8 prefill + 128 decode. `affine`은 UINT8 per-tensor activation, `u8s16c`는
INT16 activation/compute + UINT8 `new_k/new_v` storage. 각 variant는 자신의 S cache만
누적한다. 값은 step별 attention output MAE를 decode 128 step과 3 seed에 대해 평균한 것이다.

| Profile | R0K1 mean decode MAE | R1K1 mean decode MAE | 차이 | R0K1 Quantize prefill/decode | R1K1 Quantize prefill/decode |
|---|---:|---:|---:|---|---|
| affine (UINT8) | 0.03580360 | 0.03567530 | −0.00012830 | 4 / 4 (seed456: 6 / 6), 전부 RoPE 영역 | 0 / 0 |
| u8s16c | 0.01212829 | 0.01203903 | −0.00008926 | 6 / 8 (RoPE 4 + S writeout 2 / + S→C 2) | 2 / 4 |

- 실험 보고서(2026-10-29)의 R0K1 .03580361/.01212827, R1K1 .03567531/.01203907을 재현했다.
- seed별로는 u8s16c seed789에서 R1K1이 +1.75e-5 나빠졌고(affine seed별 수치는
  [report](report.md)), 128 step 중 R1이 더 나은 step은 50/4/2(u8s16c)로 대부분 step은
  출력 양자화 뒤 동일하다. 3 seed로 유의성이나 항상 개선을 주장하지 않는다.
- RoPE 사이트를 제외한 모든 observer의 dtype/qscheme/scale/zero-point는 두 variant에서
  동일했다(차이 0). CALIB가 FP를 통과시키므로 정책 변경이 공통 통계를 바꾸지 않는다.
- KV delta MAE: K는 R1에서 약간 감소(affine −4.8e-5, u8s16c −2.2e-5), V는 동일.
- R1K1 Quantize는 affine에서 0, u8s16c에서 prefill 2(S writeout)/decode 4(S→C 2 + S writeout 2).
  R0K1의 RoPE 영역 Quantize는 affine 4(seed456는 6), u8s16c 4다. 이는 실험 보고서의 관측과 같다.
- 기존 실험 subclass(`ExperimentGemma4TextAttention(rope_unified=True, kv_simplified=True)`,
  control tree에서 실행)와 production R1K1의 eager 출력/K·V delta/replay 출력은 129 step ×
  3 seed × 2 profile 모두 bit-exact 동일, observer grid도 동일했다([equiv.json](equiv.json)).
  frozen-state fingerprint만 다른데 class 이름과 extra-state 키(`gemma_rope`) 차이다.

## Saved Circle 검증 (48 artifacts)

2 variant × 2 profile × 3 seed × {prefill, decode} × {O0, O1}을 bytes로 저장하고 다시 읽어
`CircleDocument.verify()` 후 구조/ABI를 검사하고 FAKE_QUANTIZE와 NATIVE를 실행했다.

- RoPE chain(norm output → split → reordered concat): R1K1은 Q/K 모두 단일 serialized
  domain이고 영역 내 Quantize 0; R0K1은 독립 domain. 모든 CONCATENATION(RoPE, compute KV)은
  입력/출력 domain이 동일했다(#1101 legalization 유지, 48/48).
- u8s16c decode는 past S 입력 2개 각각 QUANTIZE 1개로 C 변환되고, delta 출력 2개는 S
  writeout QUANTIZE로 생성된다. 이 변환은 R1에서도 유지된다.
- 단발 strict fidelity(atol 1e-5, rtol 1e-4, ABI 인코딩 입력, eager 동일 입력): decode
  24/24 PASS. prefill은 affine seed456(K delta 7 element), u8s16c seed123(hidden 1 element),
  seed789(hidden 8 element)가 **정확히 output grid 1 step**(1.00–1.002 step) 차이로 FAIL이며
  R0K1과 R1K1에서 element 수와 크기가 동일하다. R1 고유 차이가 아니고 fp32 누산 순서에 의한
  threshold 차이로, tolerance를 늘리지 않고 FAIL로 기록한다.
- saved prefill→decode 128 step recurrent(O1): Circle이 출력한 S delta를 integer code로
  저장해 다음 decode의 S 입력에 그대로 바인딩했다. 같은 Circle cache를 받은 eager 대비 최대
  차이는 affine 0, u8s16c ≤ 1.03e-3(출력 grid 1 step 이하), FP 대비 평균 MAE는 eager
  own-cache 값과 일치(affine .03580/.03567, u8s16c .01213/.01204). 같은 S qparams에서
  K/V delta integer code mismatch는 seed별 0/1/2개로 두 variant 동일하다.
- NATIVE는 48/48 UNSUPPORTED(quantized FC integer kernel 없음). NPU latency는 측정하지 않았고
  Quantize 감소를 성능 개선으로 해석하지 않는다.
- 모든 run에서 export/evaluation 전후 weights·frozen observer state fingerprint가 동일했다.

## Wrapper smoke (`examples.inspector --mode wrapper-smoke --export circle --strict`)

11개 Gemma4 text case 모두 PASS(control tree, feature tree 각각). sliding case는 window 4,
prefill 8/decode kv 8로 window 경계를 넘는다. 저장된 Circle의 Quantize 수:

| case | control (ops, Quantize) | feature (ops, Quantize) |
|---|---|---|
| text_attention_prefill | 59, 8 | 51, 0 |
| text_attention_decode | 65, 12 | 57, 4 |
| text_attention_sliding_prefill / decode | 59, 8 / 65, 12 | 51, 0 / 57, 4 |
| text_attention_k_eq_v_prefill | 57, 8 | 49, 0 |
| text_attention_shared_kv_prefill / decode | 38, 2 / 38, 2 | 36, 0 / 36, 0 |
| text_decoder_layer_prefill / sliding_prefill | 69, 6 / 69, 6 | 63, 0 / 63, 0 |
| text_decoder_layer_decode | 75, 10 | 69, 4 |
| text_decoder_layer_shared_kv | 52, 4 | 48, 0 |

## Pretrained Gemma4 E2B (google/gemma-4-e2b-it) cached NLL

원격 photonics-a(계정별 홈 `REMOTE_HOME`), 물리 GPU index 1만 사용
(UUID `GPU-d8104d69-0e43-96d2-172b-4b6ee3224c36`를 `CUDA_VISIBLE_DEVICES`로 지정, process 안에서는
`cuda:0`; 모든 기록에 UUID/visible device를 남김). Python 3.10.12, torch 2.12.1+cu126,
transformers 5.9.0, fp32. 모델은 HF cache의 `google/gemma-4-e2b-it`
(snapshot `3e22461f65e89153144f8adb70e3b8c2cc9845a7`). 원격 worktree는 control
`1f9fac10`(clean)과 feature `6ccf4894`(clean)이며 로컬 검증 소스와 같은 커밋이다.

실행 전 고정한 protocol:

- 양자화: `gemma4_quantize.yaml` 그대로(GPTQ W4 → PTQ activation int16, weight int16,
  linear uint4, embedding/lm_head/vision patch uint8, norm int16, strict_wrap; 이전 성공 run과
  같이 `gptq.quantize_vision_patch_embed=false`). 이 recipe에서는 KV storage/compute가 모두 INT16이다.
- GPTQ는 control tree에서 **한 번만** 실행하고 변환된 모델(20.4GB)과 calibration 입력
  (vqav2 train 128 + wikitext2 train 128, seq_len 2048, decode_steps 8, 298,501 token,
  입력 sha256 `30c6768a…`)을 저장했다. 두 PTQ는 같은 파일을 읽어 weight hash
  (`d4d0e86a…`)와 calibration hash 일치를 assertion으로 확인한 뒤 fresh prepare/calibrate/convert
  했다. 양쪽 모두 GPTQ qparam 388개 재사용.
- 평가: wikitext2 **test**(calibration과 분리), chat-prefix 22 token 뒤 prefix 512 token prefill
  + 256 target을 한 토큰씩 cached decode(teacher forcing)로 32 chunk, 연속·비중첩 span,
  `evaluation.max_seq_len=2048`. 각 variant는 자신이 만든 cache를 계속 읽고 FP도 자신의 FP cache를
  쓴다. 전체 모델 hidden/logits feedback을 포함한다. 결과를 본 뒤 protocol을 바꾸지 않았다.
- scorer는 feature tree의 `cached_nll.py` 한 파일(sha256 `d2dad396…`)을 세 평가가 공유하고,
  checkpoint는 그것을 만든 tree의 `tico`로 로드했다(R0K1은 control tree).

실제 R1 활성화 증거(PTQ 기록): attention 35 layer(full 7, sliding 28; shared-KV consumer 20),
모두 `rope="pre_negated_sin"`/`unrolled`. R1K1은 Q 상속 35/35, K 상속 15/15(producer), consumer
20개는 Q만 상속, 독립 `q_x1` grid 0개. R0K1은 `_rope_domains` 없음, 독립 `q_x1` grid 35/35.
checkpoint 파일 sha256: R0K1 `81b91cd3…`, R1K1 `105c255f…`, 공유 GPTQ 모델 `067c19a2…`
(state-dict fingerprint는 [summary.json](summary.json)).

| Variant | cached target 8,192 token mean NLL | PPL | prefill-only (16,384 token) mean NLL | 대 FP |
|---|---:|---:|---:|---:|
| FP | 3.894455 | 49.129 | 4.220128 | — |
| R0K1 (control, `1f9fac10`) | 3.880916 | 48.469 | 4.217186 | −0.013539 |
| R1K1 (`6ccf4894`) | 3.880310 | 48.439 | 4.217232 | −0.014145 |

- **R1K1 − R0K1 = −0.000606 nats (PPL −0.029)**. chunk별 paired 차이는 32개 중 R1이 낮은 chunk 9,
  높은 chunk 10, 동일 13이며 범위 −0.00864 … +0.00233이다([summary.json](summary.json)
  `pretrained.cached_nll.paired`). 32 chunk로 유의성을 주장하지 않는다. W4A16 두 variant가 FP보다
  낮은 NLL을 보이는 것은 이 표본의 결과이며 일반화하지 않는다.
- feature tree에서 R0K1 checkpoint를 로드하면 `RuntimeError: Gemma4 checkpoint RoPE domain policy
  differs …`로 거부된다(실제 실행, rc=1, [validation.json](validation.json)). 자동 재해석은 없다.
- text-only 평가이며 vision 경로는 이 recipe의 vqav2 calibration으로 정상 calibrate했고 평가하지 않았다.
- 보조: 기존 `ppl` task(chat-prefix, 2048 window, stride 512, **uncached** full-window forward,
  wikitext2 test)는 FP 57.71, R0K1 56.17, R1K1 56.17(소수 둘째 자리 출력)로 past-KV 읽기 경로를
  측정하지 않으므로 KV 정책 근거가 아니라 참고값이다.

## 범위와 한계

- Tiny 결과는 random weights의 local attention 오차이며 pretrained 품질 근거가 아니다.
- triv24 UINT8→UINT8 QUANTIZE 지원은 별도 의존성이며 이번 변경은 compiler를 수정하거나
  concat legalization을 되돌리거나 dtype bridge를 추가하지 않았다. NPU compile/latency는
  미측정이다.
- 기존 R0 checkpoint(상속 구성)는 새 코드에서 거부되므로 fresh prepare/calibrate가 필요하다.
  독립 grid를 유지하려면 `q_x1/q_x2/q_cat`, `k_x1/k_x2/k_cat`에 명시 override를 둔다.

## 재실행

harness와 driver는 Git 밖(`ARTIFACT_ROOT/harness`)에 있고 manifest에 hash가 있다.

```bash
# control worktree (unmodified #1101) and feature worktree
git worktree add --detach ../TICO-rope-r0-control 1f9fac105602767678bf733f7a0ca82aa97ae5d0
# tiny matrix: run from each worktree with PYTHONPATH=<worktree>
python ARTIFACT_ROOT/harness/rope_r1_harness.py --variant R0K1 --profile u8s16c --seed 123 \
  --steps 128 --circle-steps 128 --output ARTIFACT_ROOT/matrix/u8s16c/seed123/R0K1
python ARTIFACT_ROOT/harness/compare_matrix.py --root ARTIFACT_ROOT/matrix --output ARTIFACT_ROOT/matrix-report
# focused regressions
python -m unittest test.quantization.wrapq.wrappers.gemma4.test_rope_domain_inheritance \
  test.quantization.recipes.test_cached_nll
./ccex test -k gemma4
```
