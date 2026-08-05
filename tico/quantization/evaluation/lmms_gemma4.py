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

"""Custom lmms-eval model wrapper for Gemma4.

The generic ``huggingface`` wrapper in lmms-eval passes ``audios`` (plural)
and always includes empty ``images``/``videos``/``audios`` lists to the
processor.  The Gemma4 processor expects ``audio`` (singular) and non empty image list.

This module provides a :func:`patch_huggingface_wrapper_for_gemma4` function
that monkey-patches the ``generate_until`` method of the generic
``Huggingface`` wrapper to:
  - Skip empty ``images`` / ``videos`` / ``audios`` kwargs.
  - Rename ``audios`` → ``audio`` for the Gemma4 processor.
  - Load video frames from file paths using ``decord`` and pass them as
    tensors (the Gemma4 processor does not accept video file paths).
"""

import numpy as np


def _load_video_frames(
    video_path: str,
    max_num_frames: int = 32,
) -> np.ndarray:
    """Load up to *max_num_frames* uniformly-sampled frames from a video.

    Uses ``decord`` to read the video and uniformly subsample frames.

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to return.

    Returns:
        A numpy array of shape ``(n_frames, H, W, 3)`` in RGB order.
    """
    import decord

    vr = decord.VideoReader(video_path, num_threads=1)
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video {video_path} has 0 frames.")

    if total_frames <= max_num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_num_frames, dtype=int).tolist()
        # Ensure unique indices
        indices = sorted(set(indices))

    frames = vr.get_batch(indices).asnumpy()  # (n, H, W, 3)
    return frames


def patch_huggingface_wrapper_for_gemma4(
    max_num_frames: int = 32,
):
    """Patch the generic ``Huggingface`` wrapper to work with Gemma4.

    Returns a context manager that restores the original ``generate_until``
    on exit.
    """
    import contextlib
    import time

    from lmms_eval.api.instance import GenerationResult, TokenCounts
    from lmms_eval.models.chat.huggingface import Huggingface
    from lmms_eval.protocol import ChatMessages
    from loguru import logger as eval_logger
    from tqdm import tqdm

    _original_generate_until = Huggingface.generate_until

    def _patched_generate_until(self, requests):
        res = []

        def _collate(x):
            return x[2], x[2]

        re_ords = __import__("lmms_eval.utils", fromlist=["Collator"]).Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(
            total=num_iters,
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        total_elapsed_time = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [
                doc_to_messages[0](self.task_dict[task][split][ids])
                for ids, task, split in zip(doc_id, task, split)
            ]
            chat_messages = [
                ChatMessages(**{"messages": message}) for message in chat_messages
            ]
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            batched_messages = [
                chat_message.to_hf_messages() for chat_message in chat_messages
            ]
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batched_messages
            ]

            # Extract media
            images = []
            videos = []
            audios = []
            for messages in chat_messages:
                image, video, audio = messages.extract_media()
                images.append(image)
                videos.append(video)
                audios.append(audio)
            images = self.flatten(images)
            videos = self.flatten(videos)
            audios = self.flatten(audios)

            # Load video frames from file paths into tensors.
            # The Gemma4 processor does not accept video file paths; it needs
            # video tensors (numpy arrays or torch tensors).
            video_tensors = []
            for v in videos:
                if isinstance(v, str):
                    try:
                        frames = _load_video_frames(v, max_num_frames=max_num_frames)
                        video_tensors.append(frames)
                    except Exception as e:
                        eval_logger.warning(f"Failed to load video {v}: {e}")
                else:
                    # Already a tensor/array
                    video_tensors.append(v)

            # Build kwargs for the processor.
            # The Gemma4 processor accepts: images, text, audio, videos.
            # It does NOT accept "audios" (plural).
            # Empty lists for images cause a crash in the image processor.
            kwargs = {}
            processor_kwargs = {}
            if images:
                kwargs["images"] = images
            if video_tensors:
                kwargs["videos"] = video_tensors
                processor_kwargs["videos_kwargs"] = {
                    "num_frames": max_num_frames,
                    "return_metadata": True,
                }
            if audios:
                kwargs["audio"] = audios

            inputs = self.processor(
                text=texts,
                padding=True,
                return_tensors="pt",
                **kwargs,
                **processor_kwargs,
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 4096,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_elapsed_time += end_time - start_time  # type: ignore[assignment]
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(
                    GenerationResult(
                        text=ans,
                        token_counts=TokenCounts(
                            output_tokens=len(generated_ids_trimmed[i])
                        ),
                    )
                )
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), ans
                )
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")

        res = re_ords.get_original(res)
        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": avg_speed,
        }
        __import__(
            "lmms_eval.models.model_utils.gen_metrics",
            fromlist=["log_metrics"],
        ).log_metrics(**metric_dict)

        pbar.close()
        return res

    Huggingface.generate_until = _patched_generate_until

    @contextlib.contextmanager
    def _restore():
        try:
            yield
        finally:
            Huggingface.generate_until = _original_generate_until

    return _restore()
