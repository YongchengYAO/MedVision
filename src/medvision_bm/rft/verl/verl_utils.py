from medvision_bm.sft.sft_utils import (
    _doc_to_target_TumorLesionTask,
    _doc_to_text_TumorLesionTask_CoT,
    _doc_to_text_TumorLesionTask,
    _doc_to_target_AngleDistanceTask,
    _doc_to_text_AngleDistanceTask_CoT,
    _doc_to_text_AngleDistanceTask,
    _doc_to_target_DetectionTask,
    _doc_to_text_DetectionTask_CoT,
    _doc_to_text_DetectionTask,
    format_dataset,
    clean_dataset,
    img_proccessor_nii2png_save2dataset,
    load_split_limit_dataset,
)

from medvision_bm.dataset.ds_utils import load_split_limit_dataset_tr_val_ts


def _format_data_TumorLesionTask_CoT_verl(
    example,
    model_name,
    model_hf,
    new_shape_hw=None,
):
    """
    NOTE: The function is tailored for Verl framework.

    Format data for TumorLesionTask with CoT reasoning.

    Feilds required by Verl:
        - prompt: List of messages with roles and content.
        - ground_truth: Target string.
        - data_source: Data source identifier.
        - ability: Ability identifier.
        - reward_model: Reward model information.
        - extra_info: Additional information.

    Reference:
    RLHFDataset class in Verl
    (https://github.com/YongchengYAO/verl/blob/670aeea7cd6af2de0ce7da9ae8d3fd0c522d0f0e/verl/utils/dataset/rl_dataset.py#L69)

    """
    from medvision_bm.rft.verl.rft_prompts import SYSTEM_PROMPT
    
    # Reuse existing function for SFT with CoT for TumorLesionTask
    # We can extract GT landmark coordinates from value_dict
    prompt, values_dict = _doc_to_text_TumorLesionTask_CoT(example, model_name, model_hf, new_shape_hw)
    target = _doc_to_target_TumorLesionTask(example)
    target_str = ", ".join([f"{value:.3f}" for value in target])

    # Build: "prompt"
    example["prompt"] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    # Build: "images", embedded processed image list
    example["images"] = img_proccessor_nii2png_save2dataset(example, new_shape_hw)

    # Build: "extra_info", used for medvision-tl reward
    # ---
    # Required fields:
    #   - landmark_P1_wh
    #   - landmark_P2_wh
    #   - landmark_P3_wh
    #   - landmark_P4_wh
    # Dimensions definition:
    #   - landmark_P*_wh: [relative width, relative height]
    # Note: the origin of coordinate depends on _doc_to_text_TumorLesionTask_CoT()
    # ---
    extra_info = {
        "landmark_P1_wh": [
            float(values_dict["<x1_major>"]),
            float(values_dict["<y1_major>"]),
        ],
        "landmark_P2_wh": [
            float(values_dict["<x2_major>"]),
            float(values_dict["<y2_major>"]),
        ],
        "landmark_P3_wh": [
            float(values_dict["<x1_minor>"]),
            float(values_dict["<y1_minor>"]),
        ],
        "landmark_P4_wh": [
            float(values_dict["<x2_minor>"]),
            float(values_dict["<y2_minor>"]),
        ],
    }

    # Other fields required by Verl
    example["ground_truth"] = target_str
    example["data_source"] = "medvision-tl"
    example["ability"] = "medvision-tl"
    example["reward_model"] = {"style": "rule", "ground_truth": target_str}
    example["extra_info"] = extra_info

    return example


def _format_data_TumorLesionTask_verl(
    example,
    model_name,
    model_hf,
    new_shape_hw=None,
):
    """
    NOTE: The function is tailored for Verl framework.

    Format data for TumorLesionTask with CoT reasoning.

    Feilds required by Verl:
        - prompt: List of messages with roles and content.
        - ground_truth: Target string.
        - data_source: Data source identifier.
        - ability: Ability identifier.
        - reward_model: Reward model information.
        - extra_info: Additional information.

    Reference:
    RLHFDataset class in Verl
    (https://github.com/YongchengYAO/verl/blob/670aeea7cd6af2de0ce7da9ae8d3fd0c522d0f0e/verl/utils/dataset/rl_dataset.py#L69)

    """
    from medvision_bm.rft.verl.rft_prompts import SYSTEM_PROMPT_LITE
    
    # Reuse existing function for SFT without CoT for TumorLesionTask
    prompt, _ = _doc_to_text_TumorLesionTask(example, model_name, model_hf, new_shape_hw)
    target = _doc_to_target_TumorLesionTask(example)
    target_str = ", ".join([f"{value:.3f}" for value in target])

    # Build: "prompt"
    example["prompt"] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_LITE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    # Build: "images", embedded processed image list
    example["images"] = img_proccessor_nii2png_save2dataset(example, new_shape_hw)

    # Build: "extra_info", used for reward calculation in RFT via Verl
    # ---
    # Since this function build a simple prompt without CoT,
    # and the model reasoning process in RFT (using GRPO) does not follow a CoT template,
    # we do not have process reward based on landmark coordinates here.
    # Thus, we leave extra_info empty.
    # ---
    extra_info = {"placeholder": True}

    # Other fields required by Verl
    example["ground_truth"] = target_str
    example["data_source"] = "medvision-tl"
    example["ability"] = "medvision-tl"
    example["reward_model"] = {"style": "rule", "ground_truth": target_str}
    example["extra_info"] = extra_info

    return example


def _format_data_AngleDistanceTask_CoT_verl(
    example,
    model_name,
    model_hf,
    new_shape_hw=None,
):
    """
    NOTE: The function is tailored for Verl framework.

    Format data for AngleDistanceTask with CoT reasoning.

    Feilds required by Verl:
        - prompt: List of messages with roles and content.
        - ground_truth: Target string.
        - data_source: Data source identifier.
        - ability: Ability identifier.
        - reward_model: Reward model information.
        - extra_info: Additional information.

    Reference:
    RLHFDataset class in Verl
    (https://github.com/YongchengYAO/verl/blob/670aeea7cd6af2de0ce7da9ae8d3fd0c522d0f0e/verl/utils/dataset/rl_dataset.py#L69)

    """
    from medvision_bm.rft.verl.rft_prompts import SYSTEM_PROMPT
    
    # Reuse existing function for SFT with CoT for TumorLesionTask
    # We can extract GT landmark coordinates from value_dict
    prompt, values_dict = _doc_to_text_AngleDistanceTask_CoT(example, model_name, model_hf, new_shape_hw)
    target = _doc_to_target_AngleDistanceTask(example)
    if not isinstance(target, list):
        target = [target]
    target_str = ", ".join([f"{value:.3f}" for value in target])

    # Build: "prompt"
    example["prompt"] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    # Build: "images", embedded processed image list
    example["images"] = img_proccessor_nii2png_save2dataset(example, new_shape_hw)

    # Build: "extra_info", used for medvision-ad reward
    # ---
    # Reference: https://github.com/YongchengYAO/verl/blob/medvision-rl/verl/utils/reward_score/medvision_rewards/medvision_ad.py
    # Required fields for distance metric:
    #   - metric_type
    #   - landmark_1_wh
    #   - landmark_2_wh
    # Required fields for angle metric:
    #   - metric_type
    #   - line_1_point_1_wh
    #   - line_1_point_2_wh
    #   - line_2_point_1_wh
    #   - line_2_point_2_wh
    # Dimensions definition:
    #   - *_wh: [relative width, relative height]
    # Note: the origin of coordinate depends on _doc_to_text_AngleDistanceTask_CoT()
    # ---
    metric_type = values_dict.get("metric_type", None)
    assert metric_type is not None, "metric_type not found in values_dict"

    if metric_type=="distance":
        extra_info = {
            "metric_type": "distance",
            "landmark_1_wh": [
                float(values_dict["<x1>"]),
                float(values_dict["<y1>"]),
            ], 
            "landmark_2_wh": [
                float(values_dict["<x2>"]),
                float(values_dict["<y2>"]),
            ], 
        }
    elif metric_type=="angle":
        extra_info = {
            "metric_type": "angle",
            "line_1_point_1_wh": [
                float(values_dict["<x1_line1>"]),
                float(values_dict["<y1_line1>"]),
            ], 
            "line_1_point_2_wh": [
                float(values_dict["<x2_line1>"]),
                float(values_dict["<y2_line1>"]),
            ], 
            "line_2_point_1_wh": [
                float(values_dict["<x1_line2>"]),
                float(values_dict["<y1_line2>"]),
            ], 
            "line_2_point_2_wh": [
                float(values_dict["<x2_line2>"]),
                float(values_dict["<y2_line2>"]),
            ],
        }
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    # Other fields required by Verl
    example["ground_truth"] = target_str
    example["data_source"] = "medvision-ad"
    example["ability"] = f"medvision-{metric_type}" # e.g., medvision-angle, medvision-distance
    example["reward_model"] = {"style": "rule", "ground_truth": target_str}
    example["extra_info"] = extra_info

    return example


def _format_data_AngleDistanceTask_verl(
    example,
    model_name,
    model_hf,
    new_shape_hw=None,
):
    """
    NOTE: The function is tailored for Verl framework.

    Format data for AngleDistanceTask with CoT reasoning.

    Feilds required by Verl:
        - prompt: List of messages with roles and content.
        - ground_truth: Target string.
        - data_source: Data source identifier.
        - ability: Ability identifier.
        - reward_model: Reward model information.
        - extra_info: Additional information.

    Reference:
    RLHFDataset class in Verl
    (https://github.com/YongchengYAO/verl/blob/670aeea7cd6af2de0ce7da9ae8d3fd0c522d0f0e/verl/utils/dataset/rl_dataset.py#L69)

    """
    from medvision_bm.rft.verl.rft_prompts import SYSTEM_PROMPT_LITE
    
    # Reuse existing function for SFT with CoT for TumorLesionTask
    # We can extract GT landmark coordinates from value_dict
    prompt = _doc_to_text_AngleDistanceTask(example, model_name, model_hf, new_shape_hw)
    target = _doc_to_target_AngleDistanceTask(example)
    if not isinstance(target, list):
        target = [target]
    target_str = ", ".join([f"{value:.3f}" for value in target])

    # Build: "prompt"
    example["prompt"] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_LITE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    # Build: "images", embedded processed image list
    example["images"] = img_proccessor_nii2png_save2dataset(example, new_shape_hw)

    # Extract metric_type (tailored for the MedVision dataset structure)
    biometric_profile = example["biometric_profile"]
    metric_type = biometric_profile["metric_type"]

    # Build: "extra_info", used for reward calculation in RFT via Verl
    # ---
    # Since this function build a simple prompt without CoT,
    # and the model reasoning process in RFT (using GRPO) does not follow a CoT template,
    # we do not have process reward based on landmark coordinates here.
    # Thus, we leave extra_info empty.
    # ---
    extra_info = {"metric_type": metric_type}

    # Other fields required by Verl
    example["ground_truth"] = target_str
    example["data_source"] = "medvision-ad"
    example["ability"] = f"medvision-{metric_type}" # e.g., medvision-angle, medvision-distance
    example["reward_model"] = {"style": "rule", "ground_truth": target_str}
    example["extra_info"] = extra_info

    return example


# NOTE: The arguments "model_name" and "model_hf" are not used,
# but we keep them in the function signature for consistency and future flexibility
# To check why we keep these arguments, please refer to medvision_bm/rft/verl/verl_utils/prepare_dataset_for_verl
def _format_data_DetectionTask_CoT_verl(
    example,
    model_name,
    model_hf,
    new_shape_hw=None,
):
    """
    NOTE: The function is tailored for Verl framework.

    Format data for Detection Task with CoT reasoning.

    Feilds required by Verl:
        - prompt: List of messages with roles and content.
        - ground_truth: Target string.
        - data_source: Data source identifier.
        - ability: Ability identifier.
        - reward_model: Reward model information.
        - extra_info: Additional information.

    Reference:
    RLHFDataset class in Verl
    (https://github.com/YongchengYAO/verl/blob/670aeea7cd6af2de0ce7da9ae8d3fd0c522d0f0e/verl/utils/dataset/rl_dataset.py#L69)

    """
    from medvision_bm.rft.verl.rft_prompts import SYSTEM_PROMPT

    # Reuse existing function for SFT with CoT for Detection Task
    # We can extract GT landmark coordinates from value_dict
    prompt, values_dict = _doc_to_text_DetectionTask_CoT(example)
    target = _doc_to_target_DetectionTask(example)
    if not isinstance(target, list):
        target = [target]
    target_str = ", ".join([f"{value:.3f}" for value in target])

    # Build: "prompt"
    example["prompt"] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    # Build: "images", embedded processed image list
    example["images"] = img_proccessor_nii2png_save2dataset(example, new_shape_hw)

    # Build: "extra_info", used for medvision-detection reward
    # ---
    # Required fields:
    #   - lowerleft_corner_wh
    #   - upperright_corner_wh
    # Dimensions definition:
    #   - *_corner_wh: [relative width, relative height]
    # Note: the origin of coordinate depends on _doc_to_text_DetectionTask_CoT()
    # ---
    extra_info = {
        "lowerleft_corner_wh": [
            float(values_dict["<coor0_w>"]),
            float(values_dict["<coor0_h>"]),
        ],
        "upperright_corner_wh": [
            float(values_dict["<coor1_w>"]),
            float(values_dict["<coor1_h>"]),
        ],
    }

    # Other fields required by Verl
    example["ground_truth"] = target_str
    example["data_source"] = "medvision-detection"
    example["ability"] = "medvision-detection"
    example["reward_model"] = {"style": "rule", "ground_truth": target_str}
    example["extra_info"] = extra_info

    return example


# NOTE: The arguments "model_name" and "model_hf" are not used,
# but we keep them in the function signature for consistency and future flexibility
# To check why we keep these arguments, please refer to medvision_bm/rft/verl/verl_utils/prepare_dataset_for_verl
def _format_data_DetectionTask_verl(
    example,
    model_name,
    model_hf,
    new_shape_hw=None,
):
    """
    NOTE: The function is tailored for Verl framework.

    Format data for Detection Task with CoT reasoning.

    Feilds required by Verl:
        - prompt: List of messages with roles and content.
        - ground_truth: Target string.
        - data_source: Data source identifier.
        - ability: Ability identifier.
        - reward_model: Reward model information.
        - extra_info: Additional information.

    Reference:
    RLHFDataset class in Verl
    (https://github.com/YongchengYAO/verl/blob/670aeea7cd6af2de0ce7da9ae8d3fd0c522d0f0e/verl/utils/dataset/rl_dataset.py#L69)

    """
    from medvision_bm.rft.verl.rft_prompts import SYSTEM_PROMPT_LITE

    # Reuse existing function for SFT with CoT for Detection Task
    # We can extract GT landmark coordinates from value_dict
    prompt, values_dict = _doc_to_text_DetectionTask(example)
    target = _doc_to_target_DetectionTask(example)
    if not isinstance(target, list):
        target = [target]
    target_str = ", ".join([f"{value:.3f}" for value in target])

    # Build: "prompt"
    example["prompt"] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_LITE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    # Build: "images", embedded processed image list
    example["images"] = img_proccessor_nii2png_save2dataset(example, new_shape_hw)

    # Build: "extra_info", used for medvision-detection reward
    # ---
    # Required fields:
    #   - lowerleft_corner_wh
    #   - upperright_corner_wh
    # Dimensions definition:
    #   - *_corner_wh: [relative width, relative height]
    # Note: the origin of coordinate depends on _doc_to_text_DetectionTask_CoT()
    # ---
    extra_info = {
        "lowerleft_corner_wh": [
            float(values_dict["<coor0_w>"]),
            float(values_dict["<coor0_h>"]),
        ],
        "upperright_corner_wh": [
            float(values_dict["<coor1_w>"]),
            float(values_dict["<coor1_h>"]),
        ],
    }

    # Other fields required by Verl
    example["ground_truth"] = target_str
    example["data_source"] = "medvision-detection"
    example["ability"] = "medvision-detection"
    example["reward_model"] = {"style": "rule", "ground_truth": target_str}
    example["extra_info"] = extra_info

    return example


def prepare_dataset_for_verl(
    *,
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    mapping_func,
    model_family_name,
    model_hf,
    num_workers_concat_datasets=4,
    num_workers_format_dataset=32,
    tag_ds=None,
    new_shape_hw=None,
    download_mode="reuse_dataset_if_exists",
):
    # Load and split dataset
    dataset = load_split_limit_dataset(
        tasks_list_json_path=tasks_list_json_path,
        limit_train_sample=limit_train_sample,
        limit_val_sample=limit_val_sample,
        num_workers_concat_datasets=num_workers_concat_datasets,
        tag_ds=tag_ds,
        download_mode=download_mode,
    )

    # Format dataset
    mapping_func_args = {
        "model_name": model_family_name,
        "model_hf": model_hf,
        "new_shape_hw": new_shape_hw,
    }
    # writer_batch_size=50: each worker holds at most 50 PIL images in RAM before flushing
    # to Arrow. PIL images at 512×512 RGB are ~0.75 MB each, so 50 images = ~37 MB per worker.
    # This is the key parameter for controlling peak RAM:
    #   - At writer_batch_size=1000: 128 workers × 1000 × 0.75 MB = 96 GB for buffers alone
    #   - At writer_batch_size=50:   256 workers × 50  × 0.75 MB = ~10 GB for buffers
    # Keeping this value at 50 allows scaling num_workers up to ~256 within a 200 GB RAM pod.
    dataset = format_dataset(
        dataset=dataset,
        mapping_func=mapping_func,
        mapping_func_args=mapping_func_args,
        num_workers_format_dataset=num_workers_format_dataset,
        writer_batch_size=50,
    )

    # Clean dataset to keep only necessary keys
    # ---
    # Feilds required by Verl:
    #     - prompt: List of messages with roles and content.
    #     - ground_truth: Target string.
    #     - data_source: Data source identifier.
    #     - ability: Ability identifier.
    #     - reward_model: Reward model information.
    #     - extra_info: Additional information. 
    # Additional fields:
    #     - images: the image (not just image path)
    # ---
    keys_to_keep = ["prompt", "ground_truth", "data_source", "ability", "reward_model", "extra_info", "images"]
    dataset = clean_dataset(dataset, keys_to_keep)

    return dataset


# NOTE: Test set is not used in RFT via Verl, but we prepare the dataset with test set for debugging and future flexibility.
def prepare_dataset_for_verl_with_testset(
    *,
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    mapping_func,
    model_family_name,
    model_hf,
    limit_test_sample=None,
    limit_train_sample_per_subset=None,
    limit_test_sample_per_subset=None,
    num_workers_concat_datasets=4,
    num_workers_format_dataset=32,
    tag_ds=None,
    new_shape_hw=None,
    download_mode="reuse_dataset_if_exists",
):
    # Load and split dataset
    dataset = load_split_limit_dataset_tr_val_ts(
        tasks_list_json_path=tasks_list_json_path,
        limit_train_sample=limit_train_sample,
        limit_val_sample=limit_val_sample,
        limit_test_sample=limit_test_sample,
        limit_train_sample_per_subset=limit_train_sample_per_subset,
        limit_test_sample_per_subset=limit_test_sample_per_subset,
        num_workers_concat_datasets=num_workers_concat_datasets,
        tag_ds=tag_ds,
        download_mode=download_mode,
    )

    # Format dataset
    mapping_func_args = {
        "model_name": model_family_name,
        "model_hf": model_hf,
        "new_shape_hw": new_shape_hw,
    }
    # Use small writer_batch_size because images are embedded as PIL objects (not paths).
    # Each worker buffers writer_batch_size PIL images in RAM before flushing to Arrow.
    dataset = format_dataset(
        dataset=dataset,
        mapping_func=mapping_func,
        mapping_func_args=mapping_func_args,
        num_workers_format_dataset=num_workers_format_dataset,
        writer_batch_size=50,
    )

    # Clean dataset to keep only necessary keys
    # ---
    # Feilds required by Verl:
    #     - prompt: List of messages with roles and content.
    #     - ground_truth: Target string.
    #     - data_source: Data source identifier.
    #     - ability: Ability identifier.
    #     - reward_model: Reward model information.
    #     - extra_info: Additional information. 
    # Additional fields:
    #     - images: the image (not just image path)
    # ---
    keys_to_keep = ["prompt", "ground_truth", "data_source", "ability", "reward_model", "extra_info", "images"]
    dataset = clean_dataset(dataset, keys_to_keep)

    return dataset
