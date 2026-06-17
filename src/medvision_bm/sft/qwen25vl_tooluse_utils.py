import warnings

from medvision_bm.sft.sft_prompts_tooluse import TOOL_DEF
from medvision_bm.sft.sft_utils import _doc_to_visual, mask_non_assistant_turns


def make_collate_fn_Qwen25VL_tooluse(proc):
    """Collate function for tool-use SFT with per-turn loss masking.

    Differences from make_collate_fn_Qwen25VL:
    - Applies chat template with tools=[TOOL_DEF].
    - Uses per-sample turn-level masking via mask_non_assistant_turns instead of
      global token-type masking, so only assistant turns contribute to the loss.
    """

    def _collate_fn_local(examples):
        texts = []
        images = []

        for example in examples:
            try:
                if "processed_images" in example:
                    images.append(example["processed_images"])
                elif "image_file_png" in example:
                    from PIL import Image

                    images.append(
                        [
                            Image.open(f).convert("RGB")
                            for f in example["image_file_png"]
                        ]
                    )
                elif "image_file" in example:
                    images.append(_doc_to_visual(example))
                else:
                    raise ValueError("No image field found.")

                texts.append(
                    proc.apply_chat_template(
                        example["messages"],
                        tools=[TOOL_DEF],
                        add_generation_prompt=False,
                        tokenize=False,
                    ).strip()
                )
            except (OSError, ValueError) as e:
                warnings.warn(f"Skipping example due to error: {e}")
                continue

        if not texts:
            raise RuntimeError(
                "All examples in this batch failed to process; no valid samples remain."
            )

        batch = proc(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        image_token_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
        labels[labels == proc.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100

        # Per-sample turn-level masking: mask all non-assistant turns
        for i in range(labels.shape[0]):
            labels[i] = mask_non_assistant_turns(
                batch["input_ids"][i], labels[i], proc.tokenizer
            )

        batch["labels"] = labels
        return batch

    return _collate_fn_local
