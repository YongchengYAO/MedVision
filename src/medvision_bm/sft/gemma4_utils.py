import os

from medvision_bm.sft.sft_utils import _doc_to_visual, mask_non_assistant_turns_gemma


# NOTE: This is model-specific collate function.
# Build a collate_fn bound to a specific processor (avoids relying on a global in multi-process contexts).
#
# Gemma 4 is Gemma-lineage (like Gemma 3 / MedGemma), so the base loss masking mirrors
# make_collate_fn_MedGemma: mask the padding token and the image-placeholder tokens
# (begin-of-image / end-of-image / image soft token) so the vision tokens do not
# contribute to the language-modeling loss. On top of that, when MEDVISION_SFT_COMPLETION_ONLY=1
# is set, completion-only (assistant-turn) masking is applied via mask_non_assistant_turns_gemma
# so only the model turn's response contributes to the loss -- matching make_collate_fn_Qwen25VL.
# The flag defaults OFF, keeping the legacy full-sequence objective bit-identical.
#
# Gemma 4 uses variable-resolution vision ("soft tokens"); its image tokens are still
# exposed via the tokenizer's special_tokens_map (boi/eoi/image), but to stay robust to
# any naming differences in the checkpoint we look each key up with .get() and only mask
# the ones that actually resolve to a valid id (no KeyError if a key is absent).
def make_collate_fn_Gemma4(proc):
    completion_only = os.environ.get("MEDVISION_SFT_COMPLETION_ONLY", "0") == "1"

    def _collate_fn_local(examples):
        texts = []
        images = []

        for example in examples:

            # ------------------------------
            # NOTE: image loading priority: processed_images > image_file_png (png file, load with pillow) > image_file (nii.gz file, load with _doc_to_visual)
            # ------------------------------
            try:
                if "processed_images" in example:
                    images.append(example["processed_images"])

                elif "image_file_png" in example:
                    from PIL import Image

                    pil_image = [
                        Image.open(f).convert("RGB") for f in example["image_file_png"]
                    ]
                    images.append(pil_image)

                elif "image_file" in example:
                    pil_images = _doc_to_visual(example)
                    images.append(pil_images)

                else:
                    raise ValueError(
                        "No image found in the example. Please provide 'processed_images', 'image_file_png', or 'image_file'."
                    )
                # ------------------------------

                texts.append(
                    proc.apply_chat_template(
                        example["messages"], add_generation_prompt=False, tokenize=False
                    ).strip()
                )
            except (OSError, ValueError) as e:
                # Skip examples where image loading fails
                import warnings

                warnings.warn(f"Skipping example due to image loading error: {e}")
                continue

        if not texts:
            raise RuntimeError(
                "All examples in this batch failed to process; no valid samples remain."
            )

        # Tokenize the texts and process the images
        batch = proc(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, with the padding and image tokens masked out of
        # the loss computation.
        labels = batch["input_ids"].clone()

        tokenizer = proc.tokenizer

        # Mask padding tokens
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        # NOTE: this is specific to the Gemma family (begin/end-of-image + image soft token).
        # Resolve each token defensively so a missing special-token key does not raise.
        special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
        for token_key in ("boi_token", "eoi_token", "image_token"):
            token_str = special_tokens_map.get(token_key)
            if token_str is None:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is not None and token_id >= 0:
                labels[labels == token_id] = -100

        # Opt-in completion-only masking (MEDVISION_SFT_COMPLETION_ONLY=1): mask all non-model
        # turns so only the assistant answer contributes to the loss. Default OFF keeps the
        # legacy full-sequence objective bit-identical. Safe after the pad mask because
        # pad_token_id (0) != end-of-turn (106), so the closing marker the scan keeps in the
        # loss cannot have been clobbered.
        if completion_only:
            for i in range(labels.shape[0]):
                labels[i] = mask_non_assistant_turns_gemma(
                    batch["input_ids"][i], labels[i], tokenizer
                )

        batch["labels"] = labels
        return batch

    return _collate_fn_local
