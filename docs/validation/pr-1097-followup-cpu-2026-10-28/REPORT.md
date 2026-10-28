# PR #1097 후속 CPU 구현 검증 (2026-10-28)

## 보고서 보관 범위

이 디렉터리는 baseline `b20f96d74db63db836198fa3dbe59c09b8fe822e`에서 만든
`validation/pr-1097-followup-cpu-20261028` 보고서 전용 브랜치에 보관한다.
이 보관 commit에는 `REPORT.md`와 `RESULTS.json`만 포함된다. **검증한 제품 소스·테스트
수정은 포함되지 않으므로, 이 브랜치의 HEAD 자체를 검증한 제품 코드로 해석하지 않는다.**
검증한 변경은 작업 checkout의 `fix/pr1097-followup` working tree에 보존되어 있으며,
아래 최종 코드 상태의 diff SHA256으로 식별한다.

Raw 로그·harness·`final-code.patch`·checkpoint·Circle/NPZ 등 외부 증거는 로컬
`$FOLLOWUP_ROOT`에 별도 보존되어 있고 이번 두 파일 commit에는 포함하지 않았다.
아래 실행 결과와 당시 commit/push 여부는 검증 시점의 기록이며, 이번 보고서 보관은
원래 FAIL·INCOMPLETE·strict 수치·exit code를 변경하지 않는다.

## 기준과 증거

시작 시 origin URL은 `https://github.com/mhs4670go/TICO.git`이었다.
Samsung/TICO main을 직접 fetch한 실제 baseline은
`b20f96d74db63db836198fa3dbe59c09b8fe822e` (#1097/#1098 포함)이다.
제품 변경은 이 main에서 시작했다. 과거 보고서 commit
`7cc12aab44ff87153150742fa935b4b276c492b4`는 별도 fetch/archive로 읽었고,
제품 코드에 복사/cherry-pick하지 않았다. 과거 FAIL/INCOMPLETE는 변경하지 않았다.

새 실행 증거의 루트를 `$FOLLOWUP_ROOT`로 표기한다. 이 세션의 실제 경로는
최종 응답에 제공한다. 주요 명령·환경·exit code·시간은 `harness/runs/*.json`,
stdout/stderr는 `harness/logs/*.log`에 보존했다. 추가 focused 실행은
`wrapper/*-summary.json`, `wrapper/*.log`, `rope/TASK_B_REPORT.md`에 연결했다. 기존 README와 복원 도구로
새 `harness/`에 script/config만 복원했다. 이전 결과/성공 marker는 복사하지 않았다.
관측 harness는 원래 inspector/convert를 그대로 실행한다. nested dict는 실제
signature와 serialized 이름을 대조해 바인딩하고, quantized 입력은 저장 grid에
encode한 동일 값을 Torch/reference 양쪽에 공급한 결과도 따로 기록한다.

Python 3.12.14, Torch 2.13.0+cpu, TorchVision 0.28.0+cpu,
Transformers 5.18.0, circle-schema 0.10.1, NumPy 2.5.4,
FlatBuffers 25.12.19의 기존 환경을 재사용했다. 의존성 업그레이드는 없다.
CPU, seed123(smoke)/1073(driver), OMP/MKL/OPENBLAS=1,
CUDA_VISIBLE_DEVICES="", reference runtime을 사용했다. `tico.__file__`은 실제
작업 checkout을 가리켰다. 수정 전 재현은 immutable main archive를 PYTHONPATH에
우선 배치했다. CPU driver의 git provenance에는 원본 GIT_DIR와 archive
GIT_WORK_TREE를 지정했다. 검증 실행 중 별도 worktree/commit/push/merge/PR 댓글은 만들지 않았다.

## A. Wrapper export preparation / ABI / capacity

공통 runner는 `case.export_context()`를 invocation 생성부터 실제 convert/save까지
유지한다. 기본값은 nullcontext이며 Qwen만 기존 `prepared_static_export`를 쓴다.
성공/실패 시 임시 static 상태와 원래 모드를 복원한다. generic quantization
pass/runtime를 이 smoke 준비 오류의 해결책으로 바꾸지 않았다.

- Qwen text: 모델 소유 static position/RoPE/mask preparation과 cache-free forward ABI.
  실제 serialized ABI에는 `input_ids` 외 scalar bool `use_cache`, `return_dict`도 있다.
- Qwen model/generation: production adapter에서 기존 vision/fusion/decoder/DeepStack
  adapter들을 조합한다. 고정 단일 이미지, batch1, cache-free prefill 전체 계산이다.
  vision·fusion·모든 decoder·DeepStack·norm, generation의 LM head를 유지한다.
  원래 quantized eager와 adapter 출력은 exact이며 pixels 변경도 출력에 반영된다.
- Gemma top-level: tensor-only tuple을 HF positional signature에 넘기던 결함을 제거했다.
  production `Gemma4TextLogitsPrefillExportAdapter`는 external-embedding text prefill의
  모든 decoder·norm·head·softcap을 유지한다. token lookup/PLE projection은 실제
  frozen 모델에서 host 준비한다. 기존 text-only sample의 범위이며 vision export를
  주장하지 않는다. external-embedding/cache 경로를 convert **전에 실제 calibration**한다.
  mask dict를 video 인자로 오인하는 문제를 vision calibration으로 덮지 않았다.
- Assistant: KV10을 그대로 유지하고 sliding capacity만 유효한16으로 수정했다.
  crop나 assertion 완화는 없고, capacity8의 명확한 거부는 회귀 테스트로 보존했다.

회귀 테스트는 `test/quantization/recipes/test_{qwen3_vl_wrapper_smoke_full_export,
 gemma4_top_level_smoke_export,assistant_wrapper_smoke_export_contract}.py`다.
실제 작은 모델 calibration/convert/export, 저장 파일 재로드·구조·reference 실행,
원래 eager와 adapter의 의미 동등성, source frozen state를 검증한다.
Gemma PLE 활성화/softcap과 미수집 external capability 거부, Qwen 잘못된 profile과
실패 시 context 복원도 검증한다. Assistant 수치 계약은 기존 export test의
출력-grid 4 step/rtol0이며 새로운 tolerance를 정하지 않았다.

## 후속 결과 표

아래 strict 최대 오차는 동일 encoded serialized 입력 기준이다. 저장/구조/runtime은
6개 모두 PASS이며 raw strict 결과도 [RESULTS.json](RESULTS.json)에 별도로 남겼다.
Serialized 출력은 UINT8(Qwen) 또는 INT16(Gemma/assistant), FAKE_QUANTIZE 실행 출력은
모두 FLOAT32다. 6개 plain CLI 파일은 audit 파일과 bytes가 같고 실제 재로드 출력도 exact다.

| 항목 | 수정 전 최신 main | 변경 | 수정 후 실제 결과 | 남은 제한/계약 판정 |
|---|---|---|---|---|
| `qwen3_vl_text_model` | CLI exit1, 파일 없음: static preparation 부재 | production static context/정확한 kwargs ABI | CLI exit0; 저장/구조/runtime PASS; max error [0.202556355] | full text hidden; strict FAIL 유지; 전체 출력 원인 미해결 |
| `qwen3_vl_model` | CLI exit1, 파일 없음: static positions/preparation 부재 | fixed-image full production adapter | CLI exit0; 저장/구조/runtime PASS; max error [0.452985883] | vision+text hidden; strict FAIL 유지; 전체 출력 원인 미해결 |
| `qwen3_vl_for_conditional_generation` | CLI exit1, 파일 없음: static positions/preparation 부재 | 같은 full adapter + LM head | CLI exit0; 저장/구조/runtime PASS; max error [0.076036498] | vision+text logits; strict FAIL 유지; 전체 출력 원인 미해결 |
| `gemma4_for_causal_lm` | CLI exit1, 파일 없음: float embeds가 HF input_ids로 연결 | external-embedding text logits ABI + 실제 calibration | CLI exit0; 저장/구조/runtime PASS; max error [0.000173956156] | 전체 text logits; strict FAIL 유지; 전체 출력 원인 미해결 |
| `gemma4_for_conditional_generation` | CLI exit1, 파일 없음: mask dict가 video 인자로 연결 | 같은 text logits adapter/정확한 host 입력 | CLI exit0; 저장/구조/runtime PASS; max error [7.32392073e-05] | text-only 전체 logits; vision 제외; strict FAIL 유지; 전체 출력 원인 미해결 |
| `gemma4_assistant_core` | CLI exit1, 파일 없음: KV10 > sliding capacity8 | capacity16; KV10 보존 | CLI exit0; 저장/구조/runtime PASS; max error [1.98036432e-05, 0.000246405602, 1.1280179e-05] | 기존 4-output-grid 계약 충족; strict FAIL 유지; 기존 4-grid 계약 PASS |
| RoPE 최소 재현 | domain 손실, 독립 기대값 대비 max15 | split source 인식/기존 domain/연결별 전이 보존 | max0; O0/O1, 저장·native(지원 경로)·fake 회귀 PASS | per-tensor affine 지원 범위 유지 |
| 실제 Llama attention K | strict FAIL, .0137165189 | 같은 수정 | hidden/K/V exact0, 저장/구조/runtime/frozen PASS | NPU integer 미검증 |
| 문서화된 reference 차이 | tied embedding .0841860771; Qwen MM16/token32 .00331586599 | 제품 수정 없음 | 독립 전체 출력 계산으로 GATHER table/output domain 차이 입증 | 해당 경로만 known limitation; strict FAIL 보존 |
| Frozen state | convert 후 source snapshot 보존 계약 | source 재관측/가짜 통계 없음 | 수정6개 전체 exact. 지원48개 중47개 전체 inventory exact; vision profile 1개는 기존 source 전부 exact + 독립 observer6개 추가 | 전체 inventory hash FAIL 원본도 보존; profile 추가와 source 변조 구분 |
| #1098 mixed concat | 이미 main에서 해결됨 | 중복 수정 없이 same-dtype 전이로 적용 확대 | 기존 mixed/shared/repeated/idempotent/native 회귀 유지 | uniform-input/width-direction 계약 유지 |
| 전체 numerical smoke | 과거 결과 대신 현재 registry 재확인 | case/지원 flag 삭제 없음 | 현재50/50 PASS, exit0 | plotting 없음; output-dir 파일은 생성하지 않는 기존 CLI 동작, stdout 보존 |
| 전체 supported Circle | 시도48; 저장/구조/runtime42; strict25/42 PASS | A+B | 저장/구조/runtime48/48; strict35/48 PASS | strict13 FAIL: 기존7 + 신규 export6. 미지원2는 별도 제외 |
| CPU 통합 | exit1, strict14/25 stage PASS; connected74/86 출력 PASS | B 영향 일부 개선 | exit1 보존; strict17/25, connected77/86; generation/PPL/cache/frozen PASS | 새 Qwen32 head call9 strict 차이 포함; 미해결 유지 |
| 기존/회귀 테스트 | 새 경계 테스트는 수정 전 실제 FAIL | 직접 경계 회귀 추가 | 전체4,781 실행 /4,623 PASS /158 skip /0 fail; format PASS | skip/중단을 PASS로 세지 않음 |

범위가 바뀐 Qwen/Gemma의 eager adapter 출력은 원래 검증 대상 logits/hidden 의미와
exact 비교했다. 그 의미 동등성은 저장 reference의 전체 strict parity를 보장한다는 뜻이 아니다.
13개 strict 차이 중 tied embedding만 전체 경로의 GATHER 제한으로 입증했고, assistant는
기존 grid 계약을 충족한다. 나머지11개는 미해결로 기록했다. 최대 오차/입출력 domain·shape는
[RESULTS.json](RESULTS.json)과 원본 case별 `artifact-checks.json`에 있다.

## B. RoPE split/getitem/CAT domain 전이

첫 의미 손실은 runtime이 아니라 `FoldQuantOps`에서 발생했다. affine source 탐색이
split의 getitem을 제외해 source→branch의 genuine requantization을 first annotation으로
접었다. 이어 forward는 dtype만 같으면 두 branch grid를 source로 덮어썼고,
backward CAT propagation은 x1 grid를 CAT grid로 다시 덮어썼다.

수정 범위는 네 pass다: `fold_quant_ops.py`, `propagate_qparam_forward.py`,
`propagate_qparam_backward.py`, `insert_quantize_on_dtype_mismatch.py`.
per-tensor split source를 인식하고, 확정 domain을 보존하며, 알려진 source를 가진
Quantize의 sequential rounding/clamping을 retarget으로 없애지 않는다.
#1098의 연결별 CAT 정규화를 same-dtype/distinct-grid 입력에도 재사용한다.
Uniform-input concat 기존 동작, width-direction 독립-grid 계약, per-channel/MX
제한은 유지한다. Runtime, source observer, rounding 정책은 변경하지 않았다.

실제 Llama K source는 UINT8/per-tensor `.014135867357254028`, ZP112,
shape `[1,1,16,8]`이다. torch.export의 x1/x2 Q-DQ는 각각
`.01104535162448883`/ZP143, `.013347335159778595`/ZP103,
shape `[1,1,16,4]`였다. baseline fold가 이 전이를 제거했으며,
이후 x1은 CAT `.013347335159778595`/ZP152로 해석됐다.
수정 후 split은 source grid를 유지하고 branch QUANTIZE와 필요한 CAT edge 전이를
명시적으로 저장한다. 해당 경로에 GATHER는 없고 value-preserving SPLIT/CAT
reference가 사라진 round trip을 복구하지 않는 것은 문서 계약과 일치한다.

`rope/domain-evidence.json`과 `rope/trace-{before,after}-3/trace.json`은 eager →
torch.export → fold → forward → backward → insertion → serialized/reference 경계별
실제 dtype/scale/ZP/qscheme/axis/shape/producer/consumer를 보존한다.
K x1/x2/CAT/rot intermediate max error는 baseline
`[.00543052,.20844162,.01115489,.01371652]`에서 모두0이 됐다.
동일 frozen weights/qparams/입력으로 hidden/K/V 최종 출력도 모두 exact다.

새 `test_split_concat_domain_export.py`는 수정 전 fold 및 모든8 export subcase에서
실패하고 동일-grid 대조군은 통과했다. 수정 후 asymmetric saturation/shared 및
private producer/NEG 유무/O0·O1/저장 Circle domain을 독립 기대값으로 검증한다.
지원 no-NEG 경로는 NATIVE integer ABI도 exact이며, NEG 경로는 FAKE_QUANTIZE다.
같은 grid 대조군은 불필요한 QUANTIZE가0개다. 기존 retarget E2E test의 잘못된
'중간 grid 삭제' 기대값은 sequential saturation의 독립 계산으로 강화했다.
legacy first-annotation retarget은 그대로 유지한다.

## 수치 판정과 남은 범위

저장, 구조 검사, reference 실행, raw strict 비교, encoded-input strict 비교,
계약 판정을 별도로 집계한다. strict 기준 `atol=1e-5, rtol=1e-4` 자체를 바꾸지 않았다.

`reference/`의 독립 계산은 실제 WrapQ embedding 및 actual tied-embedding case에서
reference가 dequantized table lookup을 전달하고 Torch는 별도 activation grid를
적용한다는 것을 입증했다. Tied case는 GATHER→FULLY_CONNECTED 전체를 독립 계산해
각각의 출력을 exact 재현한다. 따라서 이 사례의 `.0841860771` strict 차이는
문서화된 GATHER 제한이다. 이것을 decoder/vision/full-model 전체의 면제로 쓰지 않는다.

CPU driver Llama decoder에는 GATHER가 없다. Random source input을 serialized grid로
encode하면 clipping/first-annotation ABI 영향 일부는 줄지만, residual strict 차이는
남는다. 이는 미해결이다. Qwen multimodal embedding direct stage의 capacity16
`.00331586599`는 독립 lookup/output-roundtrip/image-splice로 양쪽을 exact 재현했다.
capacity32의 max `1.15920557e-6`는 strict 통과한다. 이 분류는 해당 direct stage에만
적용하며 connected downstream 차이는 별도로 남긴다.

검증은 tiny/synthetic CPU semantic reference 범위다. NPU integer 정확도·성능,
대형 pretrained 품질, 다른 Torch 버전은 검증하지 않았다. checkpoint/Circle/pt2/NPZ,
tensor dump/package cache는 제품 변경에 포함하지 않았다.


## CPU 통합의 상세 제한

실제 driver의110 regression +7 generation +12 cache coverage +6 evaluation =135개는
모두 PASS/skip0이다. 양쪽 revision에서 독립 PTQ한 weights/observers/modes도 exact다.
전후 각각25개, 총50개 stage artifact의 저장/구조/runtime은 성공했고,
각 artifact의 export 전후 eager stress 실행은 source frozen state를 보존했다.
실행된 eager 길이는48 token/decode4, connected capacity는16/32다.

`cpu-comparison.json`의 수정 전후 비교에서 inputs/Torch 배열은 같고 일부 Circle
배열이 달라진다. Llama254/258, Qwen16 250/262, Qwen32 238/262 배열이 exact다.
`reference/cpu-trace-deltas.json`은 첫 변화가 source observer grid를 복구한
RoPE QUANTIZE 뒤 CAT/NEG임을 입증한다. Isolated strict 판정은 개선 또는 유지됐다.

Connected 실행은 capacity별22 calls(총44), 각각43개 출력이다.16은41/43,
32는36/43 strict 통과한다. Cache valid length는12→16 및28→32, full-cache 거부와
frozen state는 유지된다. 수정된 upstream 값은 이후 실제 연결 입력도 바꾼다.
Qwen32 call9 LM head의 새 strict 차이는 max `1.7106533e-5`다. Head Circle bytes는
동일하며 두 입력을 같은 saved head와 source에 다시 공급해 재현했다.
첫 불일치는 RMS norm 한 원소의 한 grid step이다. 이는 새 입력이 노출한 기존
산술 민감도라는 경계 증거이며, 산술 원인 자체가 해결됐다는 뜻은 아니다.
GATHER 면제를 적용하지 않고 미해결로 유지한다.

## 실행 명령과 새 증거

모든 명령은 기존 `.venv`를 활성화한 checkout에서 실행했다. 아래 `$R`은
`$FOLLOWUP_ROOT`, `$C`는 작업 checkout이다. 각 run JSON에 실제 argv/cwd/환경/exit/시간이 있다.

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case all --device cpu --export none --strict --no-plot \
  --output-dir "$R/final-numerical"
# 아래 명령을 표의6개 CASE에 각각 실제 실행했다.
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case "$CASE" --device cpu --export circle --strict --no-plot \
  --output-dir "$R/final-cli/$CASE" --report-json "$R/final-cli/$CASE.json"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./ccex test -i -r reference -k gemma4.test_quant_for_
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tico.quantization.recipes.debug.cpu_validation --stage all --output "$R/final-cpu"
./ccex test
./ccex format --no-apply-patches
```

`cpu_validation --output`과 각 run artifact 디렉터리는 새 경로를 사용했다.
재실행 시 위 경로도 새 경로로 바꾼다. 소규모 재현은 추가한 regression test를 실행하면
같은 실제 production export·재로드·reference 경계를 실행한다.
matrix 수집 process의 exit0은 collection 완료만 뜻한다. 개별 strict probe의 exit1은
보존했으며, 전체 strict 집계도 `final-matrix-strict-summary` exit1로 기록했다.

`run_matrix.py`, `replay_six_cli.py`, `audit_cpu.py`와 read-only `reference/*_probe.py`는
외부 증거 디렉터리에 있으며 제품 API/대규모 검증 framework를 추가하지 않았다.

| Run / 명령 | exit | 초 | 실제 결과 |
|---|---:|---:|---|
| `pip-check` | 0 | 0.938 | dependency check PASS |
| `rope-passes-final` | 0 | 111.516 | 160 tests;158 PASS/기존 skip2; #1098 포함 |
| `gemma-offline-internal` | 0 | 82.480 | 선택22개 모두 PASS/skip0; 실제 ID는 RESULTS.json |
| `final-numerical-all` | 0 | 33.990 | 현재 registry50/50 PASS |
| `six-cli-saved-replay` | 0 | 2.269 | 6개 plain CLI 파일 재로드/구조/runtime; audit와 bytes/출력 exact |
| `baseline-cpu-validation-r2` | 1 | 325.476 | 135 tests PASS; strict14/25 stages, connected74/86 outputs; driver FAIL 유지 |
| `final-cpu-validation` | 1 | 809.253 | 135 tests PASS; strict17/25, connected77/86; driver FAIL 유지 |
| `final-full-suite-r2` | 0 | 1412.983 | 4,781 실행:4,623 PASS/158 skip/0 failures |
| `final-format` | 0 | 405.997 | 중간 전체 UFMT/Pylint/mypy PASS |
| `final-format-r2` | 0 | 215.087 | suite 완료 뒤 고정된 최종 tree 전체 UFMT/Pylint/mypy PASS |
| `final-matrix-strict-summary` | 1 | 0.521 | strict35 PASS/13 FAIL 그대로 실패 exit 반환 |
| `profile-frozen-audit` | 1 | 14.863 | raw 전체 inventory hash FAIL 보존; 기존 source 전부 exact + 독립 profile6개 추가 |
| `final-cli-qwen3_vl_text_model` | 0 | 60.091 | 실제 plain CLI 저장 성공; saved-file strict 비교는 위 표와 별도 |
| `final-cli-qwen3_vl_model` | 0 | 171.454 | 실제 plain CLI 저장 성공; saved-file strict 비교는 위 표와 별도 |
| `final-cli-qwen3_vl_for_conditional_generation` | 0 | 162.391 | 실제 plain CLI 저장 성공; saved-file strict 비교는 위 표와 별도 |
| `final-cli-gemma4_for_causal_lm` | 0 | 55.200 | 실제 plain CLI 저장 성공; saved-file strict 비교는 위 표와 별도 |
| `final-cli-gemma4_for_conditional_generation` | 0 | 41.057 | 실제 plain CLI 저장 성공; saved-file strict 비교는 위 표와 별도 |
| `final-cli-gemma4_assistant_core` | 0 | 66.277 | 실제 plain CLI 저장 성공; saved-file strict 비교는 위 표와 별도 |

기존 관련 assistant/static-input/export-adapter16개와 wrapper export-contract8개도 PASS했다.
새 Qwen 실제3 export/context, Gemma 실제2 export+PLE/capability2, assistant2 및 RoPE3개
회귀는 최종 전체 suite에서도 모두 실행·통과했다. Focused 수를 전체 suite와 더해 중복 집계하지 않았다.

전체158 skip은 기존 internal opt-in125, 미사용 Conv3d11, optional runtime/CUDA/대형
artifact/미지원 연산/선택 plotting 등의 이유다. 실제 사유는 full-suite 로그에 보존했으며
이번6개 case나 새 회귀를 skip해서 성공 수를 늘리지 않았다. Gemma internal subset의
실제22 test ID와 PASS 결과는 [RESULTS.json](RESULTS.json)의 `selected_gemma_offline_tests`에 있다.

초기 archive CPU driver1회는 git provenance 조회에서 실패했다. GIT_DIR/GIT_WORK_TREE를
명시한 새 경로의 baseline 실행으로 대체했으며 최초 exit1도 보존했다.
Qwen 신규 테스트의 첫 text replay는 실제 scalar bool 입력2개를 빠뜨린 harness 오류였다.
원본 failure를 보존하고 serialized ABI로 바로잡은 실행과 최종 suite를 따로 기록했다.
이 테스트가 수정되는 동안 먼저 시작한 전체 suite는112.399초에 중단(exit -15),
최종 suite는 `final-full-suite-r2`로 새 실행했다. 중단/초기 실패를 통과 수에 합산하지 않았다.

## 최종 코드 상태

HEAD는 시작 main 그대로이고 변경은 `fix/pr1097-followup` working tree에 있다.
검증 완료 시점에는 commit을 만들지 않았다. `tico/` + `test/`의 최종 binary diff SHA256은
`0cd1caf781c1b566b7d0c76cc91206ac9a7df4d6468cf3ab196b9b8ebd231936`이다.
외부 `final-code.patch`, `source-state.json`과 [RESULTS.json](RESULTS.json)이 이 상태를 식별한다.
수정한 책임 위치는 A의 smoke/context/production adapters + 세 recipe test와,
B의 네 pass + 세 pass test다. 사용자 동작 문서는 wrapper-smoke/export README와
`docs/design.md`의 quantization 계약을 갱신했다. 이 보고서와 작은 결과 JSON 외의
실행 산출물은 repository diff에 포함하지 않았다.
