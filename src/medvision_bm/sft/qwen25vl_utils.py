
from medvision_bm.sft.utils import _doc_to_visual
from typing import Any


# NOTE: This is model-specific collate function.
# Build a collate_fn bound to a specific processor (avoids relying on a global in multi-process contexts).
def make_collate_fn_Qwen25VL(proc):
    def _collate_fn_local(examples: list[dict[str, Any]]):
        texts = []
        images = []
        for example in examples:
            if "processed_images" in example:
                images.append(example["processed_images"])
            else:
                pil_images = _doc_to_visual(example)
                images.append(pil_images)
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