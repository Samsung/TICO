# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import random
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor


# ============================================================
# Answer normalization utilities
# - Used to reduce noise in evaluation
# - Similar to SQuAD-style normalization
# ============================================================
def normalize_answer(s: str) -> str:
    """
    Normalize an answer string by:
    - lowercasing
    - removing punctuation
    - removing articles (a, an, the)
    - collapsing multiple whitespaces

    This makes exact-match evaluation more stable.
    """
    s = s.lower().strip()

    # Treat some punctuation as word separators
    s = s.replace("-", " ").replace("/", " ")

    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)

    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(pred: str, golds: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check whether the predicted answer matches
    any of the gold answers after normalization.
    """
    p = normalize_answer(pred)
    for g in golds:
        if p == normalize_answer(g):
            return True, g
    return False, None


# ============================================================
# Dataset adapters
# - Different VQA datasets expose answers in different formats
# - These adapters convert raw samples into a unified format:
#   { image, question, golds }
# ============================================================
def get_item_vqav2(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter for VQAv2-style samples.
    """
    image = ex["image"]
    question = ex.get("question", "")

    answers = ex.get("answers", None)
    golds: List[str] = []

    if answers is None:
        golds = []
    elif isinstance(answers, dict) and "answer" in answers:
        # e.g. {"answer": [...], "confidence": [...]}
        golds = [str(a) for a in answers["answer"]]
    elif isinstance(answers, list):
        # e.g. [{"answer": "yes"}, {"answer": "yeah"}, ...]
        if len(answers) > 0 and isinstance(answers[0], dict) and "answer" in answers[0]:
            golds = [str(a["answer"]) for a in answers]
        else:
            golds = [str(a) for a in answers]
    else:
        golds = [str(answers)]

    return {"image": image, "question": question, "golds": golds}


def get_item_textvqa(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter for TextVQA-style samples.
    TextVQA is particularly sensitive to OCR degradation
    caused by quantization.
    """
    image = ex["image"]
    question = ex.get("question", "")

    answers = ex.get("answers", None)
    golds: List[str] = []

    if answers is None:
        golds = []
    elif isinstance(answers, dict) and "answer" in answers:
        golds = [str(a) for a in answers["answer"]]
    elif isinstance(answers, list):
        if len(answers) > 0 and isinstance(answers[0], dict) and "answer" in answers[0]:
            golds = [str(a["answer"]) for a in answers]
        else:
            golds = [str(a) for a in answers]
    else:
        golds = [str(answers)]

    return {"image": image, "question": question, "golds": golds}


# Mapping from dataset name to (split, adapter function)
DATASETS = {
    "vqav2": ("validation", get_item_vqav2),
    "textvqa": ("validation", get_item_textvqa),
}


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def generate_answer(
    model,
    processor,
    image,
    question: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    Qwen-VL expects image tokens in the text prompt.
    We must build the prompt via chat template with {"type":"image"}.
    """
    # 1) Build chat-style multimodal messages (image token + text)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{question}\n"
                        "Return ONLY the final answer with no extra words."
                    ),
                },
            ],
        }
    ]

    # 2) Render prompt that includes image tokens
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 3) Process multimodal inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    # Move tensors to device (some fields may be non-tensors; guard it)
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # 4) Generation kwargs
    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature  # type: ignore[assignment]

    out_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, input_len:]

    # 5) Decode
    out_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return out_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model name or path (e.g. Qwen/Qwen2.5-VL-3B-Instruct)",
    )
    ap.add_argument(
        "--dataset", type=str, choices=list(DATASETS.keys()), default="vqav2"
    )
    ap.add_argument(
        "--n", type=int, default=50, help="Number of samples for mini evaluation"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    args = ap.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    split, adapter = DATASETS[args.dataset]

    # Dataset names differ across HF mirrors.
    # Try multiple candidates to improve robustness.
    candidates = {
        "vqav2": ["HuggingFaceM4/VQAv2", "vqav2", "lmms-lab/VQAv2"],
        "textvqa": ["textvqa", "HuggingFaceM4/TextVQA", "lmms-lab/textvqa"],
    }[args.dataset]

    ds = None
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            ds = load_dataset(name, split=split, streaming=True)
            ds = ds.take(args.n)
            print(f"[info] Loaded dataset: {name} ({split}), size={len(ds)}")
            break
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(
            f"Failed to load dataset candidates={candidates}. Last error: {last_err}"
        )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    em_cnt = 0
    total = 0

    for i, ex in enumerate(ds, 1):
        item = adapter(ex)

        pred = generate_answer(
            model=model,
            processor=processor,
            image=item["image"],
            question=item["question"],
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        ok, hit = exact_match(pred, item["golds"])
        em_cnt += int(ok)
        total += 1

        # Print a few early samples and periodic updates
        if i <= 5 or i % 50 == 0:
            print("Q:", item["question"])
            print("pred:", repr(pred))
            print("pred_norm:", repr(normalize_answer(pred)))
            print("gold0:", repr(item["golds"][0] if item["golds"] else ""))
            print("golds[:10]:", [repr(x) for x in item["golds"][:10]])
            print("hit:", repr(hit))
            print("ok:", ok)
            print("-" * 60)

    print(f"\nFinal EM: {em_cnt/total:.4f}  (dataset={args.dataset}, n={total})")


if __name__ == "__main__":
    main()
