# Gemma4 cached NLL 후속 수정 (R1K1 리뷰 P2 ×2) — 2026-11-02

검토 대상은 R1K1 구현 커밋 `6ccf4894` (rebase 후 `15293554`, 내용 동일), R0K1 기준은 그 부모
`1f9fac10`(#1101 HEAD)이다. 작업은 기존 브랜치 `feat/gemma4-rope-domain-unification` 위에서
했고(사용자가 upstream `main` `1d8f2983` 위로 rebase, 내용 동일 확인), 수정 커밋은
`c6c955f9f672a8d19a9c9adc419d65b7a918b25a`이고 이 기록은 그 다음 커밋에 있다. R1 적용 조건, Q/K 독립 판정, canonical observer
소유권, freeze/checkpoint 정책, K1 storage/compute 동작은 건드리지 않았다. 변경 파일:

- `tico/quantization/recipes/evaluation/cached_nll.py`
- `tico/quantization/recipes/adapters/gemma4.py`
- `test/quantization/recipes/test_cached_nll.py`
- `tico/quantization/examples/configs/README.md`

## 결함 1 (P2): chat-prefix를 제외한 context budget

원인: adapter의 `_evaluate_cached_nll`이 `prefix_tokens + target_tokens`만
`evaluation.max_seq_len`과 비교했고, helper가 나중에 chat-prefix `H` 토큰을 앞에 붙였다.
재현(수정 전, [repro log](validation.json) `repro_before_fix`): `M=16, P=8, T=8, H=3`에서
옛 검사를 통과하고 평가가 실행되어 prefill 길이 11, 최대 cache 길이 18 > 16이었다.

해결: 계약은 **`H + P + T <= M`** 이다. `evaluate_text_cached_nll`이 chat template을 한 번
렌더링·토큰화하고(`chat_prefix_ids`), 그 `context_ids`의 실제 길이로
`validate_context_budget`을 corpus 로딩 전에 호출한다. 같은 tensor가 `evaluate_cached_nll`에
전달되어 scoring에 쓰이며 거기서도 첫 forward 전에 같은 검사를 한다. adapter는 resolve한 `M`을
`max_seq_len`으로 helper에 넘기고 자체 `P + T` 검사를 제거했다. 오류 메시지는 H, P, T, 요구 길이,
M을 포함한다. 실제 최대 누적 cache 길이는 `H + P + T - 1`(마지막 target은 다시 입력하지 않음)이지만
기존 전체 chunk 계약을 유지하고 누락된 `H`만 더했다. 이 한 토큰의 보수성은 코드 주석과 문서에
명시했고 테스트 `test_conservative_full_chunk_budget_rejects_h_plus_p_plus_t_minus_one`이 고정한다.
입력 자르기나 P/T/chunk 자동 축소, context 확대는 하지 않는다. 결과 JSON에 `max_seq_len`과
`context_token_ids`를 기록한다.

## 결함 2 (P2): dataset adapter 미적용과 빈 corpus 통과

원인: `get_dataset()`이 돌려주는 등록 adapter를 버리고 raw row의 `example.get("text", "")`를
읽었다. MMLU처럼 raw row에 `text`가 없는 등록 text dataset은 빈 문자열만 생겨 `"\n\n"` join 결과가
비어 있지 않은 줄바꿈 문자열이 되고 옛 빈 corpus 검사를 통과했다(수정 전 재현: raw MMLU 형태
2행 → corpus `'\n\n'`, tokenizer 호출 후 0 token; 빈 행 3개 → `'\n\n\n\n'` 0 token).

해결: `normalize_text_examples(dataset, adapter)`가 dataset을 **한 번만** 순회하며 등록 adapter로
각 example을 정규화하고 `text`(str)를 모은다. adapter가 callable이 아니면 `TypeError`, 정규화
결과에 str `text`가 없으면 `ValueError`(빈 문자열 대체 없음). `tokenize_text_stream`은 비공백
텍스트가 하나도 없으면 tokenizer/model 호출 전에 거부하고, 유효 텍스트가 있으면 기존대로 행을
strip하지 않고 `"\n\n"`으로 연결한다(WikiText의 빈 행 보존). `role=EVALUATION_ROLE`, split,
dataset usage 정책은 그대로다. 테스트의 `get_dataset` mock은 `adapter=None` 대신 실제 계약과 같은
callable adapter를 반환하도록 고쳤고 production에 fallback을 두지 않았다.

## 회귀 테스트 (수정 전 실패 → 수정 후 통과)

`test/quantization/recipes/test_cached_nll.py`에 추가:

| 테스트 | 검증 |
|---|---|
| `test_raw_budget_allows_prefix_plus_target_equal_to_context` | raw `P + T == M` 허용(기존 경계) |
| `test_chat_prefix_budget_allows_h_plus_p_plus_t_equal_to_context` | `H + P + T == M` 허용 |
| `test_chat_prefix_exceeding_context_is_rejected_before_any_forward` | `P + T <= M < H + P + T` 거부, 메시지에 H/P/T/요구 길이/M, forward 0회, template 토큰화 1회 |
| `test_conservative_full_chunk_budget_rejects_h_plus_p_plus_t_minus_one` | `H + P + T - 1 == M`도 거부(의도적 보수 계약), forward 0회 |
| `test_chat_prefix_is_tokenized_once_and_reused_for_scoring` | 검증에 쓴 context ids == scoring ids(`context_token_ids`), 독립 uncached NLL과 일치 |
| `test_gemma4_adapter_passes_resolved_context_to_the_helper` | adapter 진입 경로가 `M`을 전달하여 거부/허용 |
| `test_dataset_adapter_normalizes_raw_rows_before_tokenization` | raw MMLU 형태 + adapter spy, tokenizer가 adapter 텍스트 수신 |
| `test_all_empty_or_whitespace_corpus_is_rejected_before_tokenizing` | all-empty/all-whitespace 거부, tokenizer·forward 0회 |
| `test_valid_rows_with_empty_lines_keep_join_and_token_selection` | 빈 행 섞인 corpus의 join 문자열·token 선택 유지 |
| `test_adapter_output_without_text_schema_is_rejected` | text 없음/None/비mapping 거부 |

수정 전 실행: 14 tests → failures 5, errors 6(신규 9개 실패·오류, 기존 5개 통과).
수정 후: 15 tests OK. 기존 cached/uncached 정렬, chat-prefix 제외, own-cache, quantized wrapper
테스트는 그대로 유지된다.

## 실행한 검증

환경: Python 3.10.4, torch 2.13.0+cu126, transformers 5.18.0, CPU. 명령/종료 코드/로그 경로는
[validation.json](validation.json)에 있다.

| 검증 | 결과 |
|---|---|
| `python -m unittest test.quantization.recipes.test_cached_nll` (수정 전) | FAILED (failures=5, errors=6) — 의도된 재현 |
| 같은 모듈 (수정 후) | 15 OK |
| `python -m unittest` focused: `test_cached_nll`(15) + `test_evaluation_selection`(14) + `test_rope_domain_inheritance`(16) | 45 OK (같은 명령의 `test_examples_cli`는 `test/` 루트 기준 import라 unittest 직접 실행에서는 loader 오류; 아래 ccex로 실행) |
| `./ccex test -k gemma4` (R1 lifecycle/checkpoint/K1/serialized Circle O0·O1 포함) | 743 OK (skipped 70, 기존 skip) |
| `./ccex test -k examples_cli`, `-k evaluation_selection`, `-k cached_nll`, `-k dataset_usage` | 8 / 14 / 15 / 46 OK |
| `lintrunner` (변경 파일) | No lint issues |

## 기존 cached NLL 결과에 대한 영향

2026-11-02 pretrained 평가(FP/R0K1/R1K1, [이전 기록](../2026-11-02/README.md))는 다음 근거로
**유효**하며 재실행하지 않았다. 원격 CPU에서 실제 Gemma4 tokenizer/processor와 wikitext2 test
split로 옛 경로와 수정 경로를 비교했다([impact_check.json](impact_check.json)):

- 등록 adapter는 `get_item_wikitext2` = `{"text": ex.get("text", "")}`이므로 옛 raw 읽기와 같은
  텍스트를 만든다. 4,358행 중 비공백 2,891행, 옛/새 join 문자열 sha256 동일, token stream
  292,282 token 동일, 32 chunk의 target ids 동일(sha256 `893a4ecf…`).
- chat-prefix `H = 22`, `H + P + T = 790 <= M = 2048`이므로 수정된 계약으로도 통과한다.

소규모 재확인: 수정된 scorer(`cached_nll.py` sha256 `90765b51…`)로 같은 checkpoint
(R0K1 `81b91cd3…`은 control tree `1f9fac10`, R1K1 `105c255f…`은 feature tree, FP는 feature tree)
에서 원 protocol의 첫 2 chunk(512 prefill + 256 cached target, chat-prefix, M=2048)를 GPU 1
(UUID `GPU-d8104d69-0e43-96d2-172b-4b6ee3224c36`, cuda:0)에서 다시 평가했다.

| Variant (tree) | chunk 0 / chunk 1 mean NLL (수정 scorer) | 원 32-chunk run의 chunk 0 / 1 | 차이 |
|---|---|---|---|
| FP (feature tree) | 2.683270 / 3.994201 | 2.683270 / 3.994201 | 0 / 0 |
| R0K1 (control tree `1f9fac10`) | 2.708050 / 3.841833 | 2.708050 / 3.841833 | 0 / 0 |
| R1K1 (feature tree) | 2.707793 / 3.833189 | 2.707793 / 3.833189 | 0 / 0 |

세 variant 모두 512 cached target, `context_tokens=22`, `max_seq_len=2048`이 기록되었고 R0K1은
독립 `q_x1` grid 35/35, R1K1은 Q 상속 35/35·K 상속 15/15로 정책이 그대로였다. 수정된 scorer가
기존 수치를 소수점 이하 자리까지 재현하므로 기존 32-chunk 결과(FP 3.894455, R0K1 3.880916,
R1K1 3.880310; R1K1−R0K1 −0.000606 nats)는 그대로 유효하다. 2-chunk 평균은 앞 두 chunk만의
값이라 32-chunk 평균과 비교하지 않는다. 명령/로그/환경은 [validation.json](validation.json).

## 실행하지 않은 것

- 32 chunk 전체 재평가, GPTQ/PTQ 재실행, 전체 benchmark: 기존 결과가 영향받지 않아 불필요.
- triv24 UINT8→UINT8 Quantize 지원: 별도 작업.
