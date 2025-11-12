import torch

from tico.serialize.operators.adapters.llama_rmsnorm import patched_llama_rmsnorm
from tico.serialize.operators.adapters.onert.llama_attention import (
    llama_attention_forward_adapter,
)
from tico.utils.pytree_utils import register_dynamic_cache
from tico.utils.record_input import RecordingInput
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.llama.modeling_llama import LlamaAttention

from test.modules.base import TestModuleBase
from test.utils import tag


@tag.use_onert
class TinyLlamaWithFusedAttention(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.model_name = "Maykeye/TinyLLama-v0"
        self._call_count = 0
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        ).eval()
        self.fused_model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()
        for layer in self.fused_model.model.layers:
            layer.self_attn.forward = llama_attention_forward_adapter.__get__(
                layer.self_attn
            )
        self.rtol = 1e-4
        self.atol = 1e-4

    def forward(self, *args, **kwargs):
        self._call_count += 1

        if self._call_count == 2:
            return self.fused_model(*args, **kwargs)
        else:
            return self.original_model(*args, **kwargs)

    def get_example_inputs(self):
        prompt = "Lily picked up a flower."
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=30,
            truncation=True,
        )
        model = self.original_model
        model.eval()

        input_to_remove = [
            "attention_mask",
            # attention mask will be manually provided, not from example input
        ]
        condition_fn = (
            lambda args_dict: args_dict["past_key_values"].get_seq_length() != 0
        )

        with torch.no_grad(), RecordingInput(
            model, condition_fn, input_to_remove=input_to_remove
        ) as rec:
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            captured_input = rec.captured_input

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        return captured_input, {}
