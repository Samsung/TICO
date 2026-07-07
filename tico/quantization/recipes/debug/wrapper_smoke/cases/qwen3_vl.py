# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

"""Smoke cases for Qwen3-VL wrapper checks."""

from typing import Any, Mapping, Tuple

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.utils import (
    clone_module,
    first_tensor,
)


def _has_qwen3_vl() -> CaseAvailability:
    """Return availability for Hugging Face Qwen3-VL modules."""
    try:
        from tico.quantization.wrapq.utils.version import has_transformers_for

        if not has_transformers_for("qwen3-vl"):
            return CaseAvailability(
                False, "required transformers Qwen3-VL modules are unavailable"
            )
        return CaseAvailability(True)
    except Exception as exc:
        return CaseAvailability(False, f"failed to check Qwen3-VL availability: {exc}")


def _set_eager_attention(cfg: Any) -> Any:
    """Set eager attention on configs that expose a configurable implementation."""
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _make_text_config() -> Any:
    """Create a tiny Qwen3-VL text config for synthetic smoke tests."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig

    cfg = Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        attention_dropout=0.0,
        use_cache=False,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    return _set_eager_attention(cfg)


def _make_vision_config(**overrides: Any) -> Any:
    """Create a tiny Qwen3-VL vision config for synthetic smoke tests."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    params = {
        "hidden_size": 64,
        "num_heads": 4,
        "depth": 2,
        "temporal_patch_size": 2,
        "patch_size": 16,
        "out_hidden_size": 64,
        "spatial_merge_size": 2,
        "deepstack_visual_indexes": [0, 1],
    }
    params.update(overrides)
    return _set_eager_attention(Qwen3VLVisionConfig(**params))


def _make_tiny_qwen3vl_config() -> Any:
    """Create a tiny Qwen3-VL config that is large enough for image-token tests."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        vision_config={
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 2,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "out_hidden_size": 64,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0, 1],
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "num_hidden_layers": 2,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "max_position_embeddings": 1024,
            "vocab_size": 1000,
            "use_cache": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        image_token_id=998,
        video_token_id=999,
    )


def _make_tiny_qwen3vl_model() -> torch.nn.Module:
    """Build a tiny Qwen3-VL model from config without downloading weights."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    return Qwen3VLModel(_make_tiny_qwen3vl_config()).eval()


def _rope(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic rotary position embeddings."""
    emb = torch.randn(seq_len, head_dim)
    return emb.cos(), emb.sin()


def _text_rope(
    batch_size: int, seq_len: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic text RoPE embeddings."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


def _get_position_embeddings(visual_model: torch.nn.Module, grid_thw: torch.Tensor):
    """Return Qwen3-VL vision RoPE embeddings for a synthetic image grid."""
    pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
    seq_len, _ = pos_embeds.size()
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    return emb.cos(), emb.sin()


def _get_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Return cumulative sequence lengths for one synthetic Qwen3-VL image grid."""
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
    return torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)


def _make_ptq_config(
    cfg: Any, thw: Tuple[int, int, int], visual_start_idx: int = 0
) -> Any:
    """Create the PTQ config used by synthetic Qwen3-VL model examples."""
    from tico.quantization.config.ptq import PTQConfig

    return PTQConfig(
        model_args={
            "vision": {
                "grid_thw": thw,
                "visual_start_idx": visual_start_idx,
                "spatial_merge_size": cfg.vision_config.spatial_merge_size,
            }
        }
    )


def _compute_3d_position_ids(
    input_ids: torch.Tensor,
    thw: Tuple[int, int, int],
    spatial_merge_size: int,
    image_token_id: int,
) -> torch.Tensor:
    """Compute multimodal 3D RoPE position IDs for a single visual segment."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    position_ids = torch.ones(
        3, batch_size, seq_len, dtype=input_ids.dtype, device=device
    )
    for i in range(batch_size):
        image_mask = input_ids[i] == image_token_id
        image_positions = torch.nonzero(image_mask, as_tuple=True)[0]
        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        if len(image_positions) > 0:
            start_pos = image_positions[0].item()
            text_len = start_pos - st
            if text_len > 0:
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                    + st_idx
                )
            llm_grid_t = 1
            llm_grid_h = thw[1] // spatial_merge_size
            llm_grid_w = thw[2] // spatial_merge_size
            t_index = (
                torch.arange(llm_grid_t, device=device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
            num_visual_tokens = (thw[1] // spatial_merge_size) * (
                thw[2] // spatial_merge_size
            )
            st = start_pos + num_visual_tokens
        if st < seq_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = seq_len - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx
            )
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, :] = llm_positions
    return position_ids


def _create_image_input(
    cfg: Any,
    seq_len: int,
    thw: Tuple[int, int, int],
    *,
    visual_start_idx: int = 0,
    include_generation_fields: bool = False,
) -> dict[str, Any]:
    """Create one synthetic Qwen3-VL image prompt without a processor."""
    spatial_merge_size = cfg.vision_config.spatial_merge_size
    num_visual_tokens = (thw[1] // spatial_merge_size) * (thw[2] // spatial_merge_size)
    if visual_start_idx + num_visual_tokens > seq_len:
        raise ValueError("visual tokens do not fit into the synthetic sequence")
    input_ids = torch.randint(
        0, cfg.text_config.vocab_size - 2, (1, seq_len), dtype=torch.long
    )
    input_ids[
        0, visual_start_idx : visual_start_idx + num_visual_tokens
    ] = cfg.image_token_id
    pixel_values = torch.randn(
        1,
        3,
        thw[0] * cfg.vision_config.temporal_patch_size,
        thw[1] * cfg.vision_config.patch_size,
        thw[2] * cfg.vision_config.patch_size,
    )
    image_grid_thw = torch.tensor([thw])
    position_ids = _compute_3d_position_ids(
        input_ids, thw, spatial_merge_size, cfg.image_token_id
    )
    example: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": None,
        "position_ids": position_ids,
        "past_key_values": None,
        "inputs_embeds": None,
        "pixel_values": pixel_values,
        "pixel_values_videos": None,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": None,
        "cache_position": None,
    }
    if include_generation_fields:
        example["labels"] = None
        example["logits_to_keep"] = 0
    return example


def _causal_mask(seq_len: int, fill_value: float = -120.0) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), fill_value)
    return torch.triu(mask, diagonal=1)


class QwenBaseCase(WrapperSmokeCase):
    """Base class for Qwen3-VL wrapper smoke cases."""

    tags: tuple[str, ...] = ("qwen3_vl",)

    def availability(self) -> CaseAvailability:
        """Return whether this case can import Qwen3-VL modules."""
        return _has_qwen3_vl()


class QwenTextAttentionBaseCase(QwenBaseCase):
    """Base class for Qwen3-VL text attention smoke cases."""

    tags: tuple[str, ...] = ("qwen3_vl", "text", "attention")
    max_mean_abs_diff = 2.0
    seq_len = 8
    export_mode = "prefill"

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny text attention module and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        torch.manual_seed(123)
        self.text_cfg = _make_text_config()
        module = Qwen3VLTextAttention(self.text_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped text attention module in the case-specific mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        return (
            wrapped.as_export_module(self.export_mode).eval()
            if hasattr(wrapped, "as_export_module")
            else quantized
        )


class QwenTextAttentionPrefillCase(QwenTextAttentionBaseCase):
    """Smoke case for the Qwen3-VL text attention prefill wrapper path."""

    name = "qwen3_vl_text_attention_prefill"
    description = "Quantize one tiny Qwen3-VL text attention module in prefill mode."
    tags = ("qwen3_vl", "text", "attention", "prefill")
    export_mode = "prefill"

    def _sample(self) -> ForwardInput:
        """Create one synthetic prefill text attention input."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        rope = _text_rope(1, self.seq_len, self.text_cfg.head_dim)
        return ForwardInput((hidden, rope))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create prefill text attention calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the prefill text attention evaluation sample."""
        return self._sample()

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original text attention prefill signature with an explicit mask."""
        hidden, rope = sample.args
        mask = _causal_mask(hidden.shape[1])
        return reference(hidden, position_embeddings=rope, attention_mask=mask)[0]

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional prefill export inputs for the text attention adapter."""
        return ForwardInput(eval_sample.args)


class QwenTextAttentionDecodeCase(QwenTextAttentionBaseCase):
    """Smoke case for the Qwen3-VL text attention decode wrapper path."""

    name = "qwen3_vl_text_attention_decode"
    description = (
        "Quantize one tiny Qwen3-VL text attention module in static decode mode."
    )
    tags = ("qwen3_vl", "text", "attention", "decode")
    compare_reference_source = "prepared"
    export_mode = "decode"

    def _sample(self) -> ForwardInput:
        """Create one synthetic static decode text attention input."""
        hidden = torch.randn(1, 1, self.text_cfg.hidden_size)
        rope = _text_rope(1, 1, self.text_cfg.head_dim)
        mask = torch.zeros(1, 1, self.seq_len)
        past_k = torch.randn(
            1,
            self.text_cfg.num_key_value_heads,
            self.seq_len - 1,
            self.text_cfg.head_dim,
        )
        past_v = torch.randn_like(past_k)
        return ForwardInput(
            (hidden, rope),
            {
                "attention_mask": mask,
                "past_key_values": (past_k, past_v),
                "use_cache": True,
            },
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create static decode text attention calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the static decode text attention evaluation sample."""
        return self._sample()

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional static decode inputs expected by the export adapter."""
        hidden, pos = eval_sample.args
        mask = eval_sample.kwargs["attention_mask"]
        past = eval_sample.kwargs["past_key_values"]
        return ForwardInput((hidden, pos, mask, past))


class QwenTextMLPCase(QwenBaseCase):
    """Smoke case for qwen/quantize_text_mlp.py."""

    name = "qwen3_vl_text_mlp"
    description = "Quantize one tiny Qwen3-VL text MLP module."
    tags = ("qwen3_vl", "text", "mlp")
    max_mean_abs_diff = 2.0

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny text MLP module and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

        torch.manual_seed(123)
        self.text_cfg = _make_text_config()
        module = Qwen3VLTextMLP(self.text_cfg).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create text MLP calibration samples."""
        return [
            ForwardInput((torch.randn(2, 8, self.text_cfg.hidden_size),))
            for _ in range(3)
        ]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the text MLP evaluation sample."""
        return ForwardInput((torch.randn(2, 8, self.text_cfg.hidden_size),))


class QwenTextDecoderLayerCase(QwenBaseCase):
    """Smoke case for qwen/quantize_text_decoder_layer.py."""

    name = "qwen3_vl_text_decoder_layer"
    description = "Quantize one tiny Qwen3-VL text decoder layer."
    tags = ("qwen3_vl", "text", "decoder_layer")
    max_mean_abs_diff = 3.0
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny text decoder layer and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        torch.manual_seed(123)
        self.text_cfg = _make_text_config()
        module = Qwen3VLTextDecoderLayer(self.text_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic text decoder-layer input."""
        seq_len = 8
        hidden = torch.randn(1, seq_len, self.text_cfg.hidden_size)
        pos = (
            torch.randn(1, seq_len, self.text_cfg.head_dim),
            torch.randn(1, seq_len, self.text_cfg.head_dim),
        )
        mask = torch.zeros(1, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": pos,
                "attention_mask": mask,
                "position_ids": position_ids,
            },
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create text decoder-layer calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the text decoder-layer evaluation sample."""
        return self._sample()


class QwenTextModelCase(QwenBaseCase):
    """Smoke case for qwen/quantize_text_model.py."""

    name = "qwen3_vl_text_model"
    description = "Quantize a tiny Qwen3-VL text model."
    tags = ("qwen3_vl", "text", "model")
    max_mean_abs_diff = 5.0
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny text model and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        torch.manual_seed(123)
        self.text_cfg = _make_text_config()
        module = Qwen3VLTextModel(self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic text-model input."""
        ids = torch.randint(0, self.text_cfg.vocab_size, (1, 8))
        return ForwardInput((), {"input_ids": ids, "use_cache": False})

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create text-model calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the text-model evaluation sample."""
        return self._sample()

    def output_tensor(self, output: Any) -> torch.Tensor:
        """Select last_hidden_state from model outputs."""
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return first_tensor(output)


class QwenVisionMLPCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_mlp.py."""

    name = "qwen3_vl_vision_mlp"
    description = "Quantize one tiny Qwen3-VL vision MLP module."
    tags = ("qwen3_vl", "vision", "mlp")
    min_mean_abs_diff = 0.0
    max_mean_abs_diff = 1.0

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision MLP module and reference copy."""
        torch.manual_seed(123)
        model = _make_tiny_qwen3vl_model()
        self.hidden_size = model.config.vision_config.hidden_size
        module = model.visual.blocks[0].mlp.eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create vision MLP calibration samples."""
        return [ForwardInput((torch.randn(16, self.hidden_size),)) for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the vision MLP evaluation sample."""
        return ForwardInput((torch.randn(16, self.hidden_size),))


class QwenVisionAttentionCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_attention.py."""

    name = "qwen3_vl_vision_attention"
    description = "Quantize one tiny Qwen3-VL vision attention module."
    tags = ("qwen3_vl", "vision", "attention")
    min_mean_abs_diff = 0.0
    max_mean_abs_diff = 1.5

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision attention module and reference copy."""
        torch.manual_seed(123)
        model = _make_tiny_qwen3vl_model()
        self.visual = model.visual
        self.hidden_size = model.config.vision_config.hidden_size
        self.grid_thw = torch.tensor([[1, 8, 8]], dtype=torch.long)
        self.cu_seqlens = _get_cu_seqlens(self.grid_thw)
        self.position_embeddings = _get_position_embeddings(self.visual, self.grid_thw)
        self.seq_len = int(self.cu_seqlens[-1].item())
        module = model.visual.blocks[0].attn.eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic vision attention input."""
        hidden = torch.randn(self.seq_len, self.hidden_size)
        return ForwardInput((hidden, self.cu_seqlens, None, self.position_embeddings))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create vision attention calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the vision attention evaluation sample."""
        return self._sample()


class QwenVisionBlockCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_block.py."""

    name = "qwen3_vl_vision_block"
    description = "Quantize one tiny Qwen3-VL vision block."
    tags = ("qwen3_vl", "vision", "block")
    max_mean_abs_diff = 3.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision block and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config(hidden_size=64, num_heads=4)
        module = Qwen3VLVisionBlock(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic vision block input."""
        seq_len = 8
        hidden = torch.randn(seq_len, self.vision_cfg.hidden_size)
        cu_seqlens = torch.tensor([0, seq_len])
        pos = _rope(seq_len, self.vision_cfg.hidden_size // self.vision_cfg.num_heads)
        return ForwardInput((hidden, cu_seqlens), {"position_embeddings": pos})

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create vision block calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the vision block evaluation sample."""
        return self._sample()


class QwenVisionPatchEmbedCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_patch_embed.py."""

    name = "qwen3_vl_vision_patch_embed"
    description = "Quantize one tiny Qwen3-VL vision patch-embed module."
    tags = ("qwen3_vl", "vision", "patch_embed")
    max_mean_abs_diff = 3.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision patch-embed module and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed,
        )

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config(
            in_channels=3, hidden_size=32, temporal_patch_size=2, patch_size=16
        )
        module = Qwen3VLVisionPatchEmbed(self.vision_cfg).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create patch-embed calibration samples."""
        return [ForwardInput((torch.randn(1, 3, 2, 16, 16),)) for _ in range(3)]


class QwenVisionPatchMergerCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_patch_merger.py."""

    name = "qwen3_vl_vision_patch_merger"
    description = "Quantize one tiny Qwen3-VL vision patch-merger module."
    tags = ("qwen3_vl", "vision", "patch_merger")
    max_mean_abs_diff = 3.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision patch-merger module and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchMerger,
        )

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config(
            hidden_size=32, spatial_merge_size=2, out_hidden_size=64
        )
        module = Qwen3VLVisionPatchMerger(
            self.vision_cfg, use_postshuffle_norm=False
        ).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create patch-merger calibration samples."""
        return [
            ForwardInput((torch.randn(8, self.vision_cfg.hidden_size),))
            for _ in range(3)
        ]


class QwenVisionModelCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_model.py."""

    name = "qwen3_vl_vision_model"
    description = "Quantize a tiny Qwen3-VL vision model."
    tags = ("qwen3_vl", "vision", "model")
    max_mean_abs_diff = 5.0
    inplace_prepare = True
    inplace_convert = True

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the vision PTQ config with static grid metadata."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"vision": {"grid_thw": self.grid_tuple}})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision model and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config(
            depth=1, num_position_embeddings=64, deepstack_visual_indexes=[0]
        )
        self.grid_tuple = (1, 2, 2)
        self.grid_thw = torch.tensor([self.grid_tuple])
        self.sample_shape = (
            1,
            self.vision_cfg.in_channels,
            self.grid_tuple[0] * self.vision_cfg.temporal_patch_size,
            self.grid_tuple[1] * self.vision_cfg.patch_size,
            self.grid_tuple[2] * self.vision_cfg.patch_size,
        )
        module = Qwen3VLVisionModel(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic vision-model input."""
        return ForwardInput((torch.randn(self.sample_shape), self.grid_thw))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create vision-model calibration samples."""
        return [self._sample() for _ in range(2)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the vision-model evaluation sample."""
        return self._sample()

    def output_tensor(self, output: Any) -> torch.Tensor:
        """Select a stable vision-model output tensor."""
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        return first_tensor(output)


class QwenModelCase(QwenBaseCase):
    """Smoke case for qwen/quantize_model.py."""

    name = "qwen3_vl_model"
    description = "Quantize a tiny multimodal Qwen3-VL model."
    tags: tuple[str, ...] = ("qwen3_vl", "model")
    max_mean_abs_diff = 10.0
    inplace_prepare = True
    inplace_convert = True
    include_generation_fields = False

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the multimodal PTQ config with static vision metadata."""
        return _make_ptq_config(self.qwen_cfg, self.thw)

    def _model_class(self) -> type[torch.nn.Module]:
        """Return the Hugging Face model class used by this case."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        return Qwen3VLModel

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny multimodal Qwen3-VL model and reference copy."""
        torch.manual_seed(123)
        self.qwen_cfg = _make_tiny_qwen3vl_config()
        self.thw = (1, 8, 8)
        model_cls = self._model_class()
        module = model_cls(self.qwen_cfg).eval()
        return module, clone_module(module)

    def _sample(self, *, for_eval: bool = False) -> ForwardInput:
        """Create one synthetic multimodal model input."""
        sample = _create_image_input(
            self.qwen_cfg,
            seq_len=50,
            thw=self.thw,
            include_generation_fields=self.include_generation_fields,
        )
        if for_eval:
            sample = dict(sample)
            sample["position_ids"] = None
            sample["return_dict"] = False
        return ForwardInput((), sample)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create multimodal model calibration samples."""
        return [self._sample() for _ in range(2)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the multimodal model evaluation sample."""
        return self._sample(for_eval=True)


class QwenForConditionalGenerationCase(QwenModelCase):
    """Smoke case for qwen/quantize_for_conditional_generation.py."""

    name = "qwen3_vl_for_conditional_generation"
    description = "Quantize a tiny Qwen3-VL for-conditional-generation model."
    tags = ("qwen3_vl", "model", "generation")
    include_generation_fields = True

    def _model_class(self) -> type[torch.nn.Module]:
        """Return the generation model class used by this case."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        return Qwen3VLForConditionalGeneration


QWEN3_VL_CASES: tuple[WrapperSmokeCase, ...] = (
    QwenTextAttentionPrefillCase(),
    QwenTextAttentionDecodeCase(),
    QwenTextMLPCase(),
    QwenTextDecoderLayerCase(),
    QwenTextModelCase(),
    QwenVisionAttentionCase(),
    QwenVisionMLPCase(),
    QwenVisionBlockCase(),
    QwenVisionPatchEmbedCase(),
    QwenVisionPatchMergerCase(),
    QwenVisionModelCase(),
    QwenModelCase(),
    QwenForConditionalGenerationCase(),
)
