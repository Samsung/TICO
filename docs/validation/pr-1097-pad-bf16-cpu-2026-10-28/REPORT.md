# PR #1097 후속 리뷰: PAD domain / Qwen BF16 CPU 검증

## 기준·환경·기록

시작/최종 HEAD는 `335d6157803db6868d74fa250b2b9a2959beb100`, 작업 브랜치는
`fix/pr1097-followup`이다. 시작 시 origin은 `https://github.com/mhs4670go/TICO.git`이며,
실제 fetch한 원격 tip도 같은 SHA였다. `6ad0442` RoPE와 `335d615` wrapper 커밋을
그대로 보존한 위에서 추가 수정했다. 기존 보고서 두 파일의 로컬 변경도 byte hash로
보존 여부를 검사했다. 이번 변경은 미커밋 상태이며 원격 push/merge/PR 댓글은 없다.

기존 Python 3.12.14 / Torch **2.13.0+cpu** / TorchVision 0.28.0+cpu /
Transformers 5.18.0 / NumPy 2.5.4 / circle-schema 0.10.1 / FlatBuffers 25.12.19를
사용했다. `tico.__file__`은 작업 checkout을 가리킨다. 이 revision의 정책상2.13은
기본 지원 family,2.14는 qualification candidate다. 기존2.13 CPU 환경이 정상이라
의존성 설치·업그레이드 없이 수정 전후를 검증했다. `pip check`도 통과했다.

이 문서의 `$R`은 새 외부 증거 디렉터리 `$REVIEW_ROOT`다. 세션의 실제 경로는 최종
응답에 제공한다. `runs/*.json`에 argv/cwd/CPU 환경/원래 exit code/시간,
`logs/*.log`에 stdout/stderr를 보존했다. OMP/MKL/OPENBLAS=1,
CUDA_VISIBLE_DEVICES="", 기본 reference runtime을 사용했다.
Checkpoint/Circle/NPZ·대형 tensor dump는 제품 diff에 포함하지 않았다.

## P1 — zero PAD 뒤 genuine requantization

수정 파일:

- `tico/quantization/passes/fold_quant_ops.py`
- 신규 `test/quantization/passes/test_pad_domain_export.py`
- 계약 문서 `docs/design.md`

원인은 PAD의 affine source-domain 상속 제외였다. Q(B)가 PAD output annotation으로
접히면 native PAD가 복사한 A의 integer code를 B로 해석했다. 공통 forward guard와
runtime의 raw-code-copy 동작은 그대로 유지하고, **per-tensor affine zero PAD만**
알려진 source A를 상속시켰다. 따라서 PAD input/output은 A이며 필요한 Q(B)는 실제
QUANTIZE로 남는다. B metadata를 A로 바꿔 결과를 숨기는 수정이 아니다.

아래 모든 재현은 `tico.convert → 저장 → 재로드 → verify(raise_on_error=True) →
NATIVE 및 FAKE_QUANTIZE`를 O0/O1에서 실제 실행했다. Storage/NATIVE dtype은 UINT8,
FAKE 출력은 FLOAT32다. 모든 input/output domain은 per-tensor affine, range0..255다.

| 사례 | Serialized input A / output B (전후 동일) | 수정 전 NATIVE | 수정 후 NATIVE | 수정 후 dequantized / FAKE |
|---|---|---|---|---|
| 최초, shape[1,4] | A=.5/ZP127; B=1/ZP0; input codes `[129,131]` | `[0,129,131,0]` FAIL | `[0,1,2,0]` exact | `[0,1,2,0]` exact |
| Rounding+saturation, shape[1,5] | A=.125/ZP128; B=.5/ZP3; input codes `[111,137,151]` | `[3,111,137,151,3]` FAIL | `[3,0,5,9,3]` exact | `[0,-1.5,1,3,0]` exact |
| Same-grid, shape[1,5] | A=B=.125/ZP128; input codes `[111,137,151]` | `[128,111,137,151,128]` exact | 同左 exact | `[0,-2.125,1.125,2.875,0]` exact |

최초 사례의 수정 전 FAKE는 이미 맞았지만, rounding/saturation 사례의 수정 전 FAKE는
`[0,-2.125,1.125,2.875,0]`로 B의 실제 round trip을 누락했다. 수정 후 두 mode 모두
독립적인 상수 기대값과 일치한다. Expected를 converter/runtime/helper 출력으로 만들지 않았다.

수정 전 서로 다른 domain에서도 operator는 PAD뿐이었다. 수정 후에는 PAD(A→A)와
QUANTIZE(A→B) 하나다. 같은 grid에서는 QUANTIZE가0개다. Shared 변형 `(result, source)`의
source A qparams와 원래 source integer codes도 그대로다. 회귀6 methods는8개 실제
saved O0/O1 exports를 수행하며, 별도 before/after probe는 각각12개 변형을 실행했다.
Nonzero padding 거부와 per-channel source 상속 제한도 대조군으로 검증한다.

수정 전 새 회귀는6 subtest failure(exit1), 수정 후6 methods 모두 PASS다.
`pad/{before,after}/result.json`, `pad/P1_SUMMARY.json`,
`logs/pad-regression-{before,after}.log`에 원래 결과를 보존했다.

## P2 — static mask/RoPE를 실제 hidden에 임시 정렬

수정 파일:

- `tico/quantization/wrapq/wrappers/qwen_vl/export_adapters.py`
- 기존 `test/quantization/recipes/test_qwen3_vl_wrapper_smoke_full_export.py`
- 계약 문서 `tico/quantization/recipes/export/README.md`

원래 eager static 경로는 mask/RoPE를 hidden dtype/device로 맞추지만 새 전체 prefill
adapter는 FP32 static 값을 그대로 fake-quantize했다. BF16 Q/K가 FP32로 승격되어
BF16 V와의 `attn_i @ v_i`에서 오류가 났다. mask/cos/sin의 **임시 `.to(hidden)` 값**만
`_fq()`에 전달하도록 수정했다. 준비된 buffer, model dtype, frozen grid는 재할당하지 않는다.
Position IDs는 INT64로 유지한다.

재현은 실제 tiny model을 BF16으로 만든 뒤 prepare하고, calibration sample의 floating
leaf만 BF16으로 바꿔 실제 calibration했다. 이후 **convert 전에** 실제 관측으로
`derive_candidate()`한 per-channel grid를 `load_qparams(candidate.scale.float(),
candidate.zero_point, lock=True)`로 import했다. 모든 해당 frozen scale은 FP32다.
미관측 observer를 건너뛰거나 가짜 통계를 넣지 않았다. Inference `.eval()` 상태를
고정한 뒤 원래 training/quantization mode와 frozen source를 snapshot했다.

| Case | 수정 전 eager | 수정 전 adapter | 수정 후 eager/adapter |
|---|---|---|---|
| QwenModelCase | BF16 [1,50,64], PASS | float != c10::BFloat16 오류 | BF16 [1,50,64], max error0, atol0/rtol0 exact |
| QwenForConditionalGenerationCase | BF16 [1,50,1000], PASS | 같은 matmul 오류 | BF16 [1,50,1000], max error0, atol0/rtol0 exact |

Static mask/cos/sin은 실행 전후 FLOAT32/CPU, position IDs는 INT64/CPU다. Tensor identity,
값·dtype·device가 그대로이며, 양쪽 case의 source weights/observer snapshot/mode도
정상 및 강제 예외 context 후 동일하다. 임시 `_static_export_inputs` 등록이 정상/예외
경로 모두 제거됨을 검사한다. 최종 회귀는 정상 종료 직후 training mode도 별도로 비교한다.

최종 canonical-eval 회귀를 immutable335d615 production package에 실행하면1 method의
두 case 모두 dtype error(exit1)로 실패했다. 수정 후 동일 계산·state 계약은 PASS다.
원본 fixture의 첫 after 실행은 exact 출력은 맞았지만 fresh wrapper의 training flag를
inference eval로 정규화하기 전 snapshot한 테스트 오류였다. 원본 exit1도 보존했고,
`.eval()` 후 snapshot하면서 전체 preservation assertion을 유지했다.

기존 observer compatibility는10 run/9 PASS/기존 CUDA skip1이다. BF16 **per-channel
scale**의 CPU 실행 거부와 BF16 activation+FP32 scale 허용 테스트는 실제 실행·통과했다.
이번 결과는 BF16 adapter forward 범위이며 BF16 Circle capture/serialization/runtime
지원이나 NPU 정확도·성능을 검증했다는 뜻이 아니다. BF16 Circle은 추가 실행하지 않았다.

## 관련 검증 및 최종 gate

| Run (정확한 argv는 동일 이름의 `runs/*.json`) | exit | 초 | 결과 |
|---|---:|---:|---|
| `pip-check` | 0 | 0.415 | 환경 의존성 PASS |
| `pad-regression-before` | 1 | 7.717 | 6 methods 중6 subtest failure; source/B domain 오류를 실제 integer 실행으로 검출 |
| `pad-regression-after` | 0 | 5.476 | 6 methods PASS;8개 saved O0/O1 exports + 부적용 대조군 |
| `pad-related-passes` | 0 | 28.489 | 166 run /164 PASS /기존 slice skip2; 요청한4개 모듈·width/MX 포함 |
| `pad-native-runtime-corrected` | 0 | 1.720 | 39 run /39 PASS /skip0 |
| `qwen-bf16-final-regression-before` | 1 | 6.189 | 최종 inference fixture +335d615 production:1 method/errors2 |
| `qwen-bf16-final-mode-regression` | 0 | 6.229 | 1 method/2개 BF16 case PASS; 정상·예외 source/static/mode 보존 exact |
| `observer-execution-compatibility` | 0 | 1.820 | 10 run /9 PASS /기존 CUDA skip1; CPU BF16 scale 거부 실제 실행 |
| `wrapper-export-related` | 0 | 138.546 | 8 methods PASS; Qwen3/Gemma2/assistant1 실제 FP32 저장·재로드·구조·reference 실행 |
| `final-full-suite` | 0 | 1212.865 | 4788 run /4630 PASS /기존 skip158 /failure0 |
| `final-format` | 0 | 219.038 | 고정된 최종 소스 UFMT/Pylint/mypy PASS; No lint issues; patch 미적용 |

전체 suite의158 skip은 기존 internal opt-in·미지원 kernel/operator·optional runtime/CUDA·대형
artifact 등의 사유다. 새 PAD6개와 BF161개 methods는 최종 suite에서 실제 실행·통과했다.
Focused 수를 전체 suite에 더해 중복 합산하지 않았다. 원래 FAIL·skip·harness 오류 로그는 그대로 보존했다.

실행 명령의 정확한 원문은 [RESULTS.json](RESULTS.json)과 외부 `runs/*.json`에 있다.
핵심 명령은 다음과 같다. 모두 기존 `.venv`와 위 CPU 환경을 사용했다.

```bash
python -m unittest test.quantization.passes.test_pad_domain_export -v
python -m unittest discover -s test/quantization/passes -v
python -m unittest test.unit_test.circle.runtime.test_quantized_execution \
  test.unit_test.circle.runtime.test_kernels_shape.ConcatSplitSliceTest.test_pad_and_pad_v2 -v
python -m unittest \
  test.quantization.recipes.test_qwen3_vl_wrapper_smoke_full_export.TestQwen3VLWrapperSmokeFullExport.test_bfloat16_full_prefill_matches_eager_and_preserves_static_inputs -v
python -m unittest test.quantization.wrapq.observers.test_execution_compatibility -v
# Existing FP32 Qwen two methods + Gemma top-level4 + assistant2, exact argv in wrapper-export-related.json.
./ccex test
./ccex format --no-apply-patches
```

처음 native/runtime 명령은 존재하지 않는 `ShapeKernelTest`를 지정한 harness 오류로
exit1이었다.38개 실제 test PASS와1개 lookup error를 전체 PASS로 세지 않았고,
올바른 `ConcatSplitSliceTest`로 실행한 별도39 PASS 결과를 사용했다.

## 보존 계약·남은 범위

RoPE split/getitem source domain, genuine sequential rounding/saturation,
shared producer/consumer, CAT repeated edge/mixed dtype/width-direction,
MX axis/block, affine per-channel 제한 및 zero-only padding 범위를 유지했다.
공통 propagation guard·runtime·serializer·Gemma/assistant 제품 코드는 변경하지 않았다.
Tolerance/skip/지원 flag를 바꾸지 않았다. 과거 보고서의 unresolved strict 및 CPU driver
exit1 기록은 당시 결과로 보존한다. 이번 작업에서 CPU driver나 전체48-case 수치 matrix를
다시 실행하지 않았고, P1/P2 성공을 그 미해결 항목의 성공으로 집계하지 않았다.

Torch2.14.1 candidate·CUDA/NPU·외부 checkpoint/데이터셋은 실행하지 않았다.
기본 지원2.13에서 두 리뷰 재현이 모두 성립했으므로 별도 의존성 환경을 만들지 않았다.
새 표의 저장·구조·runtime PASS와 수치 PASS는 각 단계별 증거이며 target backend 보장은 아니다.

## 최종 소스 식별과 증거 위치

최종 source는 HEAD335d615 +6개 제품/테스트/계약 문서 변경이다. 신규 PAD test를
`git add -N`으로 포함한 뒤 명시적인6개 경로에 `git diff --binary HEAD -- <paths>`를
실행했다. 이 patch bytes의 SHA256은
`a5bad7c6c76e0f4e446e6db6c7dcefeba7210aad388dc1bb129a3174234ce91b`이다.
각 파일 SHA256과 명시적 경로 목록은 [RESULTS.json](RESULTS.json)의 `final_source`에 있다.
원본 `$R/final-source.patch`와 `$R/final-source.json`으로 미커밋 상태를 재현할 수 있다.
기존 보고서 로컬 파일은 이 patch에 포함하지 않았으며 최초 hash와 동일함을 확인했다.

최종 gate 시작 전에 source를 고정했고, 두 gate 후 patch·각 파일·기존 보고서 hash가
모두 동일함을 다시 확인했다. Gate 진행 중 실행 소스 변경은 없었다.
새 검증 기록 두 파일 자체는 실행 소스 patch fingerprint의 대상에서 제외한다.
외부 원본은 `$R/logs`, `$R/runs`, `$R/pad`, `$R/qwen`이며 실제 절대 경로는 최종 응답에 있다.
