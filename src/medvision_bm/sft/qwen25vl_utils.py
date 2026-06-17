from medvision_bm.sft.sft_utils import _doc_to_visual, mask_non_assistant_turns


# NOTE: This is model-specific collate function.
# Build a collate_fn bound to a specific processor (avoids relying on a global in multi-process contexts).
def make_collate_fn_Qwen25VL(proc):
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
            except (OSError, ValueError, Exception) as e:
                # Skip examples where image loading fails
                import warnings

                warnings.warn(f"Skipping example due to image loading error: {e}")
                continue

        batch = proc(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        image_token_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
        labels[labels == proc.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100

        # Per-sample turn-level masking: mask all non-assistant turns (system/user)
        # so only assistant answers contribute to the loss. Without this the entire
        # user prompt (incl. the long pixel-size arithmetic text) was trained on,
        # diluting the loss. Mirrors make_collate_fn_Qwen25VL_tooluse.
        for i in range(labels.shape[0]):
            labels[i] = mask_non_assistant_turns(
                batch["input_ids"][i], labels[i], proc.tokenizer
            )

        batch["labels"] = labels
        return batch

    return _collate_fn_local
