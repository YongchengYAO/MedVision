from medvision_bm.sft.utils import _doc_to_visual


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
            if "processed_images" in example:
                images.append(example["processed_images"])

            elif "image_file_png" in example:
                from PIL import Image
                pil_image = [Image.open(f).convert("RGB")
                             for f in example["image_file_png"]]
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

        batch = proc(text=texts, images=images,
                     return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        image_token_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
        image_begin_token_id = [
            proc.tokenizer.convert_tokens_to_ids("<|im_start|>")]
        image_end_token_id = [
            proc.tokenizer.convert_tokens_to_ids("<|im_end|>")]

        labels[labels == proc.tokenizer.pad_token_id] = -100
        labels[labels == image_begin_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == image_end_token_id] = -100

        batch["labels"] = labels
        return batch
    return _collate_fn_local
