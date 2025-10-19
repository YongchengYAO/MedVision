from medvision_bm.sft.utils import _doc_to_visual


# NOTE: This is model-specific collate function.
# Build a collate_fn bound to a specific processor (avoids relying on a global in multi-process contexts).
def make_collate_fn_MedGemma(proc):
    def _collate_fn_local(examples):
        texts = []
        images = []

        for example in examples:

            # ------------------------------
            # NOTE: image loading priority: processed_images > image_file_png (png file, load with pillow) > image_file (nii.gz file, load with _doc_to_visual)
            # ------------------------------
            if "processed_images" in example:
                images.append(example["processed_images"])

            elif "image_file_png" in example:
                from PIL import Image
                pil_image = Image.open(
                    example["image_file_png"]).convert("RGB")
                images.append(pil_image)

            elif "image_file" in example:
                pil_images = _doc_to_visual(example)
                images.append(pil_images)

            else:
                raise ValueError(
                    "No image found in the example. Please provide 'processed_images', 'image_file_png', or 'image_file'.")
            # ------------------------------

            texts.append(
                proc.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                ).strip()
            )

        # Tokenize the texts and process the images
        batch = proc(text=texts, images=images,
                     return_tensors="pt", padding=True)

        # The labels are the input_ids, with the padding and image tokens masked in
        # the loss computation
        labels = batch["input_ids"].clone()

        # NOTE: this is specific to the MedGemma model
        # Image tokens
        begin_of_image_token_id = [
            proc.tokenizer.convert_tokens_to_ids(
                proc.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        end_of_image_token_id = [
            proc.tokenizer.convert_tokens_to_ids(
                proc.tokenizer.special_tokens_map["eoi_token"]
            )
        ]
        image_token_id = [
            proc.tokenizer.convert_tokens_to_ids(
                proc.tokenizer.special_tokens_map["image_token"]
            )
        ]
        # Mask tokens that are not used in the loss computation
        labels[labels == proc.tokenizer.pad_token_id] = -100
        labels[labels == begin_of_image_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == end_of_image_token_id] = -100

        batch["labels"] = labels
        return batch

    return _collate_fn_local
