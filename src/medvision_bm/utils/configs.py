"""Shared constants for the MedVision benchmark.

This module centralizes the configuration values used across evaluation,
summarization, and analysis scripts. It defines:

- The global random ``SEED`` for reproducibility.
- Summary output filenames for the tumor/lesion (TL), angle/distance (AD), and
  detection tasks.
- Thresholds and grouping parameters (e.g. near-zero ground-truth cutoff,
  minimum group size, excluded and tumor/lesion group keys, random-box
  simulation count).
- Dataset mappings, notably ``DATASETS_NAME2PACKAGE`` (dataset name to importable
  package name) and the list of tasks that force standard image normalization.
- Anatomy label maps: ``label_map_regroup`` (fine-grained labels to coarse
  anatomy groups) and ``label_map_rename`` (fine-grained labels to canonical
  names).
- CT windowing presets: ``HU_window_WL_map`` (window width/level presets) and
  ``CT_HU_windows_WL`` (per anatomy group to a windowing preset).

Note:
    Do not rename the variables defined here; they are imported by name
    elsewhere in the codebase.
"""

# NOTE: Do not change the variable names in this file, as they are imported elsewhere
#

# Random seed for reproducibility, widly used across the codebase
SEED = 1024


# ----------------------------------------------------------------
# NOTE: Summary filename in T/L tasks
# ----------------------------------------------------------------
# Mainly used in summarize_TL_task.py
SUMMARY_FILENAME_TL_METRICS = "summary_metrics_TL_Task.json"
SUMMARY_FILENAME_TL_VALUES = "summary_values_TL_Task.json"
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# NOTE: Summary filename in A/D tasks
# ----------------------------------------------------------------
# Mainly used in summarize_AD_task.py
SUMMARY_FILENAME_AD_METRICS = "summary_metrics_AD_Task.json"
SUMMARY_FILENAME_AD_VALUES = "summary_values_AD_Task.json"
# Samples whose ground-truth angle/distance is below this threshold are excluded
# from metric aggregation — near-zero GT causes unbounded MRE.
AD_NEAR_ZERO_GT_THRESHOLD = 0.1
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# NOTE: Summary filename in detection tasks
# ----------------------------------------------------------------
# Mainly used in summarize_detection_task.py
SUMMARY_FILENAME_DETECT_METRICS = "summary_metrics_detect_Task.json"
SUMMARY_FILENAME_DETECT_VALUES = "summary_values_detect_Task.json"
SUMMARY_FILENAME_GROUPED_ANATOMY_VS_TUMOR_LESION_DETECT_METRICS = (
    "summary_metrics_anatomy_vs_lesion_detect_Task.json"
)
SUMMARY_FILENAME_ALL_MODELS_DETECT_METRICS = (
    "summary_metrics_all_models_detect_Task.json"
)

# Used in analyze_detection_task_boxsize_vs_random.py
SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES = (
    "summary_values_per_boxImgRatio_detect_Task.json"
)
SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS = (
    "summary_metrics_per_boxImgRatio_detect_Task.json"
)

# Used in analyze_detection_task_boxsize.py
# Anatomy-level: fine-grained labels collapsed via label_map_regroup → "<AnatomyGroup> @ <Modality>"
SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_METRICS = (
    "summary_metrics_per_sample_detect_Task.csv"
)
SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_MEAN_METRICS = (
    "summary_metrics_boxImgRatio_x_label_detect_Task.csv"
)
# Label-level: raw fine-grained labels kept as-is → "<label_name> @ <Modality>"
SUMMARY_FILENAME_PER_BOX_IMG_RATIO_FINELABEL_DETECT_METRICS = (
    "summary_metrics_per_sample_fineLabel_detect_Task.csv"
)
SUMMARY_FILENAME_PER_BOX_IMG_RATIO_FINELABEL_DETECT_MEAN_METRICS = (
    "summary_metrics_boxImgRatio_x_fineLabel_detect_Task.csv"
)
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# NOTE: These constants are mainly used in summarize_detection_task.py
# ----------------------------------------------------------------
# Minimum sample size for a label to be included in the group average calculation (anatomy and Tumor/Lesion groups)
MINIMUM_GROUP_SIZE = 50
# Keys to be excluded from group calculations
EXCLUDED_KEYS = ["miscellaneous", "others"]
# Keywords indicating Tumor/Lesion group labels
TUMOR_LESION_GROUP_KEYS = ["tumor", "lesion", "metastatic"]
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# NOTE: Used in analyze_detection_task_boxsize_vs_random.py
# ----------------------------------------------------------------
# Number of random box simulations for random detection model
RANDOM_BOX_SIMULATIONS = 100
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# NOTE: keep this mapping updated when new datasets are added
# ----------------------------------------------------------------
# Mapping from dataset names to package names
# e.g., "AbdomenAtlas1.0Mini" -> "AbdomenAtlas__1_0__Mini"
# Package names is used for module import:
# e.g., from medvision_ds.datasets.AbdomenAtlas__1_0__Mini import preprocess_detection, preprocess_segmentation
DATASETS_NAME2PACKAGE = {
    "ACDC": "ACDC",
    "AFIDs": "AFIDs",
    "AMOS22": "AMOS22",
    "AbdomenAtlas1.0Mini": "AbdomenAtlas__1_0__Mini",
    "AbdomenCT-1K": "AbdomenCT_1K",
    "BCV15": "BCV15",
    "BraTS24": "BraTS24",
    "CAMUS": "CAMUS",
    "Ceph-Biometrics-400": "Ceph_Biometrics_400",
    "CrossMoDA": "CrossMoDA",
    "DEEP-PSMA": "DEEP_PSMA",
    "FLARE22": "FLARE22",
    "FeTA24": "FeTA24",
    "HNTSMRG24": "HNTSMRG24",
    "ISLES24": "ISLES24",
    "KiPA22": "KiPA22",
    "KiTS23": "KiTS23",
    "LIDC-IDRI": "LIDC_IDRI",
    "LNQ2023": "LNQ2023",
    "MAMA-MIA": "MAMA_MIA",
    "MSD": "MSD",
    "OAIZIB-CM": "OAIZIB_CM",
    "PDDCA": "PDDCA",
    "PI-CAI": "PICAI",
    "SKM-TEA": "SKM_TEA",
    "ToothFairy2": "ToothFairy2",
    "TopCoW24": "TopCoW24",
    "TotalSegmentator": "TotalSegmentator",
    "VerSe": "VerSe",
    "autoPET-III": "autoPET_III",
}
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# NOTE: A list of dictionaries specifying tasks that require standard image normalization.
# This is mainly designed to skip HU-based CT image normalization for contrast CT scans, such as KiPA22.
TASK_LIST_FORCE_STANDARD_IMAGE_NORMALIZATION = [
    {
        "dataset_name": "KiPA22",
        "taskID": "01",
        "taskType": "Tumor-Lesion-Size",
    },  # TL task
    {
        "dataset_name": "KiPA22",
        "taskID": "01",
        "taskType": "Box-Size",
    },  # Detection task
    {
        "dataset_name": "KiPA22",
        "taskID": "01",
        "taskType": "Mask-Size",
    },  # Mask-Size task
]
# ----------------------------------------------------------------


label_map_regroup = {
    # ───────────────────────────── VASCULATURE ─────────────────────────────
    # arteries
    "aorta": "Artery",
    "anterior communicating artery": "Artery",
    "basilar artery": "Artery",
    "left iliac artery": "Artery",
    "left anterior cerebral artery": "Artery",
    "left common carotid artery": "Artery",
    "left internal carotid artery": "Artery",
    "left middle cerebral artery": "Artery",
    "left posterior cerebral artery": "Artery",
    "left posterior communicating artery": "Artery",
    "left subclavian artery": "Artery",
    "renal artery": "Artery",
    "right iliac artery": "Artery",
    "right anterior cerebral artery": "Artery",
    "right common carotid artery": "Artery",
    "right internal carotid artery": "Artery",
    "right middle cerebral artery": "Artery",
    "right posterior cerebral artery": "Artery",
    "right posterior communicating artery": "Artery",
    "right subclavian artery": "Artery",
    "third a2 segment": "Artery",
    "brachiocephalic trunk": "Artery",
    # veins
    "superior vena cava": "Vein",
    "inferior vena cava": "Vein",
    "inferior vena cava (ivc)": "Vein",
    "postcava": "Vein",
    "postcava (inferior vena cava)": "Vein",
    "portal and splenic veins": "Vein",
    "portal vein and splenic vein": "Vein",
    "left brachiocephalic vein": "Vein",
    "right brachiocephalic vein": "Vein",
    "left iliac vein": "Vein",
    "right iliac vein": "Vein",
    "renal vein": "Vein",
    # ───────────────────────────── BRAIN ─────────────────────────────
    # brain structures
    "brain": "Brain",
    "skull": "Brain",
    "anterior hippocampus": "Brain",
    "posterior hippocampus": "Brain",
    "deep grey matter": "Brain",
    "grey matter": "Brain",
    "white matter": "Brain",
    "brainstem": "Brain",
    "cerebellum": "Brain",
    "ventricles": "Brain",
    "external cerebrospinal fluid": "Brain",
    # brain lesions and tumors
    "stroke infarct": "Brain Tumor/Lesion",
    "peritumoral edema of brain": "Brain Tumor/Lesion",
    "edema of brain": "Brain Tumor/Lesion",
    "surrounding non-enhancing flair hyperintensity of brain": "Brain Tumor/Lesion",
    "resection cavity of brain": "Brain Tumor/Lesion",
    "enhancing brain tumor": "Brain Tumor/Lesion",
    "enhancing brain tumor tissue": "Brain Tumor/Lesion",
    "non-enhancing brain tumor": "Brain Tumor/Lesion",
    "non-enhancing brain tumor core": "Brain Tumor/Lesion",
    "gross tumor volume of brain": "Brain Tumor/Lesion",
    "cystic component of brain": "Brain Tumor/Lesion",
    # ───────────────────────────── HEART ─────────────────────────────
    "heart": "Heart",
    "left atrium": "Heart",
    "left atrium of heart": "Heart",
    "left atrial appendage": "Heart",
    "left ventricular cavity": "Heart",
    "left ventricular myocardium": "Heart",
    "left ventricle": "Heart",
    "right ventricular cavity": "Heart",
    "myocardium": "Heart",
    # ───────────────────────────── THORAX – LUNGS & PLEURA ───────────
    "left lung": "Lung",
    "left lung lower lobe": "Lung",
    "left lung upper lobe": "Lung",
    "right lung": "Lung",
    "right lung lower lobe": "Lung",
    "right lung middle lobe": "Lung",
    "right lung upper lobe": "Lung",
    "lung cancer": "Lung Tumor/Lesion",
    "lung nodule": "Lung Tumor/Lesion",
    # ───────────────────────────── ABDOMINAL ORGANS ────────────────
    # liver
    "liver": "Liver",
    "liver vessel": "Liver",
    "liver cancer": "Liver Tumor/Lesion",
    "liver tumour": "Liver Tumor/Lesion",
    "liver tumor": "Liver Tumor/Lesion",
    # kidneys
    "kidney": "Kidney",
    "right kidney": "Kidney",
    "left kidney": "Kidney",
    "kidney cyst": "Kidney Tumor/Lesion",
    "left kidney cyst": "Kidney Tumor/Lesion",
    "right kidney cyst": "Kidney Tumor/Lesion",
    "kidney tumor": "Kidney Tumor/Lesion",
    # pancreas
    "pancreas": "Pancreas",
    "pancreas cancer": "Pancreas Tumor/Lesion",
    # gallbladder
    "gall bladder": "Gallbladder",
    "gallbladder": "Gallbladder",
    # spleen
    "spleen": "Spleen",
    # adrenal glands
    "adrenal gland": "Adrenal Gland",  # generic term kept for completeness
    "left adrenal gland": "Adrenal Gland",
    "left adrenal gland (lag)": "Adrenal Gland",
    "right adrenal gland": "Adrenal Gland",
    "right adrenal gland (rag)": "Adrenal Gland",
    # colon
    "colon": "Colon",
    "colon cancer primaries": "Colon Tumor/Lesion",
    # intestines
    "rectum": "Intestine",
    "duodenum": "Intestine",
    "small bowel": "Intestine",
    "esophagus": "Esophagus",
    # stomach
    "stomach": "Stomach",
    # ───────────────────────────── URO-GYNAE ──────────────────────────
    # urinary system
    "urinary bladder": "Urinary System",
    "bladder": "Urinary System",
    # uterus
    "uterus": "Uterus",
    # prostate
    "prostate": "Prostate",
    "peripheral zone of prostate": "Prostate",
    "transition zone of prostate": "Prostate",
    "clinically significant prostate cancer lesion": "Prostate Tumor/Lesion",
    # breast
    "breast tumor": "Breast Tumor/Lesion",
    # ───────────────────────────── THROAT & AIRWAY ───────────────────
    # head & neck
    "cochlea": "Head-Neck",
    "trachea": "Head-Neck",
    "pharynx": "Head-Neck",
    "thyroid gland": "Head-Neck",
    "left parotid gland": "Head-Neck",
    "right parotid gland": "Head-Neck",
    "left submandibular gland": "Head-Neck",
    "right submandibular gland": "Head-Neck",
    # optic apparatus (PDDCA organs at risk); the chiasm stays with the nerves it belongs to
    "left optic nerve": "Head-Neck",
    "right optic nerve": "Head-Neck",
    "optic chiasm": "Head-Neck",
    "primary gross tumor volume (head & neck)": "Head-Neck Tumor/Lesion",
    "vestibular schwannoma": "Head-Neck Tumor/Lesion",
    # ───────────────────────────── MUSCULOSKELETAL ───────────────────
    # hip
    "left hip": "Hip",
    "right hip": "Hip",
    "sacrum": "Hip",
    "left gluteus maximus": "Hip",
    "left gluteus medius": "Hip",
    "left gluteus minimus": "Hip",
    "right gluteus maximus": "Hip",
    "right gluteus medius": "Hip",
    "right gluteus minimus": "Hip",
    "right iliopsoas": "Hip",
    "left iliopsoas": "Hip",
    # ribs
    "left 1st rib": "Rib",
    "left 2nd rib": "Rib",
    "left 3rd rib": "Rib",
    "right 1st rib": "Rib",
    "right 2nd rib": "Rib",
    "right 3rd rib": "Rib",
    **{f"{side} {n}th rib": "Rib" for side in ("left", "right") for n in range(4, 13)},
    "costal cartilages": "Rib",
    # spine
    **{
        f"vertebra {lvl}": "Spine"
        for lvl in (
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6",
            "T7",
            "T8",
            "T9",
            "T10",
            "T11",
            "T12",
            "T13",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L6",
            "S1",
        )
    },
    "vertebrae": "Spine",
    "intervertebral discs": "Spine",
    "spinal cord": "Spine",
    # knee
    "femur": "Knee Bone",
    "tibia": "Knee Bone",
    "left femur": "Knee Bone",
    "right femur": "Knee Bone",
    "femoral cartilage": "Knee Soft Tissue",
    "lateral tibial cartilage": "Knee Soft Tissue",
    "medial tibial cartilage": "Knee Soft Tissue",
    "patellar cartilage": "Knee Soft Tissue",
    "lateral meniscus": "Knee Soft Tissue",
    "medial meniscus": "Knee Soft Tissue",
    # lymphatics
    "metastatic lymph node": "Metastatic Lymph Node",
    "mediastinal lymph node": "Metastatic Lymph Node",
    # miscellaneous pathology (non-organ specific)
    "edema": "Miscellaneous Tumor/Lesion",
    "tumor": "Miscellaneous Tumor/Lesion",
    "cystic component": "Miscellaneous Tumor/Lesion",
    # whole-body PET tumour burden — deliberately not tied to one organ
    "total tumor burden": "Miscellaneous Tumor/Lesion",
    # dentistry
    "upper jawbone": "Jawbone",
    "lower jawbone": "Jawbone",
    "mandible": "Jawbone",
    "left inferior alveolar canal": "Jawbone",
    "right inferior alveolar canal": "Jawbone",
    # teeth
    **{
        t: "Tooth"
        for t in [
            "upper left canine",
            "upper left central incisor",
            "upper left lateral incisor",
            "upper left first premolar",
            "upper left second premolar",
            "upper left first molar",
            "upper left second molar",
            "upper left third molar (wisdom tooth)",
            "upper right canine",
            "upper right central incisor",
            "upper right lateral incisor",
            "upper right first premolar",
            "upper right second premolar",
            "upper right first molar",
            "upper right second molar",
            "upper right third molar (wisdom tooth)",
            "lower left canine",
            "lower left central incisor",
            "lower left lateral incisor",
            "lower left first premolar",
            "lower left second premolar",
            "lower left first molar",
            "lower left second molar",
            "lower left third molar (wisdom tooth)",
            "lower right canine",
            "lower right central incisor",
            "lower right lateral incisor",
            "lower right first premolar",
            "lower right second premolar",
            "lower right first molar",
            "lower right second molar",
            "lower right third molar (wisdom tooth)",
        ]
    },
    # catch-alls
    "na": "Others",
    "implant": "Others",
    "crown": "Others",
    "bridge": "Others",
    "left autochthon": "Others",
    "right autochthon": "Others",
    "sternum": "Others",
    "humerus": "Others",
    "left humerus": "Others",
    "right humerus": "Others",
    "left clavicle": "Others",
    "right clavicle": "Others",
    "left scapula": "Others",
    "right scapula": "Others",
    "prostate/uterus": "Others",
    # ── genuinely-missing anatomy (medvision_ds uses these names; absent from the lists above) ──
    "pulmonary vein": "Vein",
    "left maxillary sinus": "Head-Neck",  # no sinus-specific group; Head-Neck is closest
    "right maxillary sinus": "Head-Neck",
}


label_map_rename = {
    # ───────────────────────────── VASCULATURE ─────────────────────────────
    # arteries
    "aorta": "aorta",
    "anterior communicating artery": "anterior communicating artery",
    "basilar artery": "basilar artery",
    "left iliac artery": "left iliac artery",
    "left anterior cerebral artery": "left anterior cerebral artery",
    "left common carotid artery": "left common carotid artery",
    "left internal carotid artery": "left internal carotid artery",
    "left middle cerebral artery": "left middle cerebral artery",
    "left posterior cerebral artery": "left posterior cerebral artery",
    "left posterior communicating artery": "left posterior communicating artery",
    "left subclavian artery": "left subclavian artery",
    "renal artery": "renal artery",
    "right iliac artery": "right iliac artery",
    "right anterior cerebral artery": "right anterior cerebral artery",
    "right common carotid artery": "right common carotid artery",
    "right internal carotid artery": "right internal carotid artery",
    "right middle cerebral artery": "right middle cerebral artery",
    "right posterior cerebral artery": "right posterior cerebral artery",
    "right posterior communicating artery": "right posterior communicating artery",
    "right subclavian artery": "right subclavian artery",
    "third a2 segment": "third a2 segment",
    "brachiocephalic trunk": "brachiocephalic trunk",
    # veins
    "superior vena cava": "superior vena cava",
    "inferior vena cava": "inferior vena cava",
    "inferior vena cava (ivc)": "inferior vena cava",
    "postcava": "postcava",
    "postcava (inferior vena cava)": "postcava",
    "portal and splenic veins": "portal and splenic veins",
    "portal vein and splenic vein": "portal and splenic vein",
    "left brachiocephalic vein": "left brachiocephalic vein",
    "right brachiocephalic vein": "right brachiocephalic vein",
    "left iliac vein": "left iliac vein",
    "right iliac vein": "right iliac vein",
    "renal vein": "renal vein",
    # ───────────────────────────── BRAIN ─────────────────────────────
    # brain structures
    "brain": "brain",
    "skull": "skull",
    "anterior hippocampus": "hippocampus",
    "posterior hippocampus": "hippocampus",
    "deep grey matter": "grey matter",
    "grey matter": "grey matter",
    "white matter": "white matter",
    "brainstem": "brainstem",
    "cerebellum": "cerebellum",
    "ventricles": "ventricles",
    "external cerebrospinal fluid": "cerebrospinal fluid",
    # brain lesions and tumors
    "stroke infarct": "stroke infarct",
    "peritumoral edema of brain": "brain edema",
    "edema of brain": "brain edema",
    "surrounding non-enhancing flair hyperintensity of brain": "non-enhancing flair hyperintensity of brain",
    "resection cavity of brain": "brain resection cavity",
    "enhancing brain tumor": "enhancing brain tumor",
    "enhancing brain tumor tissue": "enhancing brain tumor",
    "non-enhancing brain tumor": "non-enhancing brain tumor",
    "non-enhancing brain tumor core": "non-enhancing brain tumor core",
    "gross tumor volume of brain": "brain tumor",
    "cystic component of brain": "brain cyst",
    # ───────────────────────────── HEART ─────────────────────────────
    "heart": "heart",
    "left atrium": "left atrium",
    "left atrium of heart": "left atrium",
    "left atrial appendage": "left atrial appendage",
    "left ventricular cavity": "left ventricular cavity",
    "left ventricular myocardium": "left ventricular myocardium",
    "left ventricle": "left ventricle",
    "right ventricular cavity": "right ventricular cavity",
    "myocardium": "myocardium",
    # ───────────────────────────── THORAX – LUNGS & PLEURA ───────────
    "left lung": "left lung",
    "left lung lower lobe": "left lung lower lobe",
    "left lung upper lobe": "left lung upper lobe",
    "right lung": "right lung",
    "right lung lower lobe": "right lung lower lobe",
    "right lung middle lobe": "right lung middle lobe",
    "right lung upper lobe": "right lung upper lobe",
    "lung cancer": "lung cancer",
    # ───────────────────────────── ABDOMINAL ORGANS ────────────────
    # liver
    "liver": "liver",
    "liver vessel": "liver vessel",
    "liver cancer": "liver tumor",
    "liver tumor": "liver tumor",
    "liver tumour": "liver tumor",
    # kidneys
    "kidney": "kidney",
    "right kidney": "right kidney",
    "left kidney": "left kidney",
    "kidney cyst": "kidney cyst",
    "left kidney cyst": "left kidney cyst",
    "right kidney cyst": "right kidney cyst",
    "kidney tumor": "kidney tumor",
    # pancreas
    "pancreas": "pancreas",
    "pancreas cancer": "pancreas cancer",
    # gallbladder
    "gall bladder": "gallbladder",
    "gallbladder": "gallbladder",
    # spleen
    "spleen": "spleen",
    # adrenal glands
    "adrenal gland": "adrenal gland",
    "left adrenal gland": "left adrenal gland",
    "left adrenal gland (lag)": "left adrenal gland",
    "right adrenal gland": "right adrenal gland",
    "right adrenal gland (rag)": "right adrenal gland",
    # colon
    "colon": "colon",
    "colon cancer primaries": "colon cancer",
    # intestines
    "rectum": "rectum",
    "duodenum": "duodenum",
    "small bowel": "small bowel",
    "esophagus": "esophagus",
    # stomach
    "stomach": "stomach",
    # ───────────────────────────── URO-GYNAE ──────────────────────────
    # urinary system
    "urinary bladder": "urinary bladder",
    "bladder": "bladder",
    # uterus
    "uterus": "uterus",
    # prostate
    "prostate": "prostate",
    "peripheral zone of prostate": "prostate",
    "transition zone of prostate": "prostate",
    # ambiguous
    "prostate/uterus": "prostate/uterus",
    # ───────────────────────────── THROAT & AIRWAY ───────────────────
    # head & neck
    "vestibular schwannoma": "vestibular schwannoma",  # this is a tumor
    "cochlea": "cochlea",
    "trachea": "trachea",
    "pharynx": "pharynx",
    "thyroid gland": "thyroid gland",
    "primary gross tumor volume (head & neck)": "head & neck tumor",
    # ───────────────────────────── MUSCULOSKELETAL ───────────────────
    # hip
    "left hip": "left hip",
    "right hip": "right hip",
    "sacrum": "sacrum",
    "left gluteus maximus": "left gluteus maximus",
    "left gluteus medius": "left gluteus medius",
    "left gluteus minimus": "left gluteus minimus",
    "right gluteus maximus": "right gluteus maximus",
    "right gluteus medius": "right gluteus medius",
    "right gluteus minimus": "right gluteus minimus",
    "right iliopsoas": "right iliopsoas",
    "left iliopsoas": "left iliopsoas",
    # ribs
    "left 1st rib": "left 1st rib",
    "left 2nd rib": "left 2nd rib",
    "left 3rd rib": "left 3rd rib",
    "right 1st rib": "right 1st rib",
    "right 2nd rib": "right 2nd rib",
    "right 3rd rib": "right 3rd rib",
    **{
        f"{side} {n}th rib": f"{side} {n}th rib"
        for side in ("left", "right")
        for n in range(4, 13)
    },
    "costal cartilages": "costal cartilages",
    # spine
    **{
        f"vertebra {lvl}": f"vertebra {lvl}"
        for lvl in (
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6",
            "T7",
            "T8",
            "T9",
            "T10",
            "T11",
            "T12",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "S1",
        )
    },
    "vertebrae": "vertebrae",
    "intervertebral discs": "intervertebral discs",
    "spinal cord": "spinal cord",
    # knee
    "femur": "femur",
    "tibia": "tibia",
    "left femur": "left femur",
    "right femur": "right femur",
    "femoral cartilage": "femoral cartilage",
    "lateral tibial cartilage": "lateral tibial cartilage",
    "medial tibial cartilage": "medial tibial cartilage",
    "patellar cartilage": "patellar cartilage",
    "lateral meniscus": "lateral meniscus",
    "medial meniscus": "medial meniscus",
    # lymphatics
    "metastatic lymph node": "metastatic lymph node",
    # miscellaneous pathology (non-organ specific)
    "edema": "miscellaneous tumor/lesion",
    "tumor": "miscellaneous tumor/lesion",
    "cystic component": "miscellaneous tumor/lesion",
    # dentistry
    "upper jawbone": "upper jawbone",
    "lower jawbone": "lower jawbone",
    "left inferior alveolar canal": "left inferior alveolar canal",
    "right inferior alveolar canal": "right inferior alveolar canal",
    # teeth
    **{
        t: t
        for t in [
            "upper left canine",
            "upper left central incisor",
            "upper left lateral incisor",
            "upper left first premolar",
            "upper left second premolar",
            "upper left first molar",
            "upper left second molar",
            "upper left third molar (wisdom tooth)",
            "upper right canine",
            "upper right central incisor",
            "upper right lateral incisor",
            "upper right first premolar",
            "upper right second premolar",
            "upper right first molar",
            "upper right second molar",
            "upper right third molar (wisdom tooth)",
            "lower left canine",
            "lower left central incisor",
            "lower left lateral incisor",
            "lower left first premolar",
            "lower left second premolar",
            "lower left first molar",
            "lower left second molar",
            "lower left third molar (wisdom tooth)",
            "lower right canine",
            "lower right central incisor",
            "lower right lateral incisor",
            "lower right first premolar",
            "lower right second premolar",
            "lower right first molar",
            "lower right second molar",
            "lower right third molar (wisdom tooth)",
        ]
    },
    # catch-alls
    "na": "others",
    "implant": "others",
    "crown": "others",
    "bridge": "others",
    "left autochthon": "left autochthon",
    "right autochthon": "right autochthon",
    "sternum": "sternum",
    "humerus": "humerus",
    "left humerus": "left humerus",
    "right humerus": "right humerus",
    "left clavicle": "left clavicle",
    "right clavicle": "right clavicle",
    "left scapula": "left scapula",
    "right scapula": "right scapula",
}


HU_window_WL_map = {
    "soft_tissue": (400, 40),
    "lung": (1500, -600),
    "brain": (80, 40),
    "bone": (1800, 400),
}

CT_HU_windows_WL = {
    "Artery": HU_window_WL_map["soft_tissue"],
    "Vein": HU_window_WL_map["soft_tissue"],
    "Brain": HU_window_WL_map["brain"],
    "Brain Tumor/Lesion": HU_window_WL_map["brain"],
    "Heart": HU_window_WL_map["soft_tissue"],
    "Lung": HU_window_WL_map["lung"],
    "Lung Tumor/Lesion": HU_window_WL_map["lung"],
    "Liver": HU_window_WL_map["soft_tissue"],
    "Liver Tumor/Lesion": HU_window_WL_map["soft_tissue"],
    "Kidney": HU_window_WL_map["soft_tissue"],
    "Kidney Tumor/Lesion": HU_window_WL_map["soft_tissue"],
    "Pancreas": HU_window_WL_map["soft_tissue"],
    "Pancreas Tumor/Lesion": HU_window_WL_map["soft_tissue"],
    "Gallbladder": HU_window_WL_map["soft_tissue"],
    "Spleen": HU_window_WL_map["soft_tissue"],
    "Adrenal Gland": HU_window_WL_map["soft_tissue"],
    "Colon": HU_window_WL_map["soft_tissue"],
    "Colon Tumor/Lesion": HU_window_WL_map["soft_tissue"],
    "Intestine": HU_window_WL_map["soft_tissue"],
    "Esophagus": HU_window_WL_map["soft_tissue"],
    "Stomach": HU_window_WL_map["soft_tissue"],
    "Urinary System": HU_window_WL_map["soft_tissue"],
    "Uterus": HU_window_WL_map["soft_tissue"],
    "Prostate": HU_window_WL_map["soft_tissue"],
    # PI-CAI and MAMA-MIA are MRI-only; the entries keep this table total over anatomy groups
    "Prostate Tumor/Lesion": HU_window_WL_map["soft_tissue"],
    "Breast Tumor/Lesion": HU_window_WL_map["soft_tissue"],
    "Head-Neck": HU_window_WL_map["soft_tissue"],
    "Head-Neck Tumor/Lesion": HU_window_WL_map["soft_tissue"],
    "Hip": HU_window_WL_map["bone"],
    "Rib": HU_window_WL_map["bone"],
    "Spine": HU_window_WL_map["bone"],
    "Knee Bone": HU_window_WL_map["bone"],
    "Knee Soft Tissue": HU_window_WL_map["soft_tissue"],
    "Metastatic Lymph Node": HU_window_WL_map["soft_tissue"],
    "Miscellaneous Tumor/Lesion": HU_window_WL_map["soft_tissue"],
    "Jawbone": HU_window_WL_map["bone"],
    "Tooth": HU_window_WL_map["bone"],
}


# ----------------------------------------------------------------
# NOTE: Dataset donut-chart palettes
# ----------------------------------------------------------------
# Muted qualitative hue-sweeps, one colour per dataset, used by
# summarize_datasets.py. When there are more datasets than colours the list is
# cycled and each wrap is blended further toward white (later rounds lighter);
# colours are assigned in ring (annotation-count) order so the palette sweeps
# the donut in listed sequence. Select the active palette in
# summarize_datasets.py (``_DATASET_COLORS``).
nature_palette_1 = [
    "#B66699", "#D49AB5", "#B7A6C7", "#A3BDD8", "#8EBCBB",
    "#85B293", "#B9C18E", "#E8C38C", "#E8A27D",
]
nature_palette_2 = [
    "#9F8DB8", "#6E8FB2", "#ABC8E5","#7DA494", "#D0D08A",
    "#EAB67A", "#E5A79A", "#C16E71", "#D8A0C1",
]
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# NOTE: Palette cycling for more targets than colours
# ----------------------------------------------------------------
# When a palette runs out, ``extend_palette`` reuses each base hue but shifts it
# hard in HLS so the repeat stays visually separable both from its own base and
# from every other colour in the list. Wraps alternate dark / light around the
# base with growing magnitude, and each wrap also rotates hue:
#
#   wrap 0  pure base
#   wrap 1  darker  + hue -20 deg
#   wrap 2  lighter, saturation boosted + hue +20 deg
#   wrap 3  darker still + hue -40 deg
#   ...
#
# Three details matter:
#
#   * Lightness moves as a fraction of the remaining headroom to black / white,
#     not as a flat offset. A flat offset washes out on an already-pale palette
#     (nature_palette_2), which is why the old lighten-only cycling produced
#     near-identical wedges.
#   * The lighter wrap BOOSTS saturation. Lightening alone drags every hue
#     toward white, collapsing distinct base hues into the same pale grey; the
#     boost keeps them hued.
#   * _WRAP_LIGHTNESS_MIN floors how dark a wrap may go. Chasing maximum dE
#     alone drove the darker wraps to near-black (L* 17-25), which reads as
#     muddy rather than as a distinct colour. The floor trades some separation
#     for wrapped colours that stay mid-tone (L* >= ~32).
#
# Minimum pairwise dE, old lighten-only cycling -> this function:
#   nature_palette_1, 22 targets:   3.0 ->  9.8
#   nature_palette_2, 22 targets:   5.3 -> 10.7
#   radar_model_colors, 16 targets:17.9 -> 12.6
# (dE ~2.3 is a just-noticeable difference, so the old 3.0 was effectively a
# duplicate.) Cycling a short palette this far is inherently lossy -- prefer a
# longer base palette when more than ~2 wraps are expected.
_WRAP_LIGHTNESS_BLEND = (0.30, 0.46, 0.56)  # per magnitude step; last value repeats
_WRAP_LIGHT_SATURATION = 1.40               # saturation scale on lighter wraps
_WRAP_HUE_STEP = 20.0 / 360.0               # +/- per magnitude step
_WRAP_LIGHTNESS_MIN = 0.34                  # floor: keep wrapped colours off black
_WRAP_LIGHTNESS_MAX = 0.93                  # ceiling: keep wrapped colours off white


def _wrap_shift(wrap):
    """(lightness_blend, saturation_scale, hue_offset) for cycle round ``wrap``.

    ``wrap`` 0 is the untouched base. Odd wraps go darker, even wraps go lighter,
    and the magnitude grows every two wraps so no two rounds land on the same
    colour. ``lightness_blend`` is a signed fraction of the headroom toward black
    (negative) or white (positive).
    """
    if wrap <= 0:
        return 0.0, 1.0, 0.0
    step = (wrap + 1) // 2                      # 1, 1, 2, 2, 3, 3, ...
    darker = wrap % 2 == 1                      # wrap 1 dark, wrap 2 light, ...
    blend = _WRAP_LIGHTNESS_BLEND[min(step - 1, len(_WRAP_LIGHTNESS_BLEND) - 1)]
    if darker:
        return -blend, 1.0, -_WRAP_HUE_STEP * step
    return blend, _WRAP_LIGHT_SATURATION, _WRAP_HUE_STEP * step


def extend_palette(base_colors, n, as_hex=True):
    """``n`` visually distinct colours cycled from ``base_colors``.

    The first ``len(base_colors)`` entries are the base palette unchanged; every
    further wrap re-uses the same hues with the HLS shift described above, so a
    target list longer than the palette never produces two identical colours.

    Args:
        base_colors: palette to cycle. Each entry may be a ``#RRGGBB`` string or
            an RGB(A) tuple of floats in 0..1 (e.g. ``plt.cm.tab10.colors``).
        n: number of colours to return.
        as_hex: return ``#RRGGBB`` strings (default) instead of RGB float tuples.

    Returns:
        list of length ``n``.
    """
    import colorsys

    if not base_colors:
        raise ValueError("base_colors must not be empty")

    period = len(base_colors)
    out = []
    for i in range(n):
        base = base_colors[i % period]
        if isinstance(base, str):
            hex_str = base.lstrip("#")
            rgb = tuple(int(hex_str[j : j + 2], 16) / 255.0 for j in (0, 2, 4))
        else:
            rgb = tuple(float(c) for c in base[:3])

        d_light, s_scale, d_hue = _wrap_shift(i // period)
        if d_light == 0.0 and s_scale == 1.0 and d_hue == 0.0:
            out.append(_rgb_to_hex(rgb) if as_hex else rgb)
            continue

        h, l, s = colorsys.rgb_to_hls(*rgb)
        # blend toward black (d_light < 0) or white (d_light > 0) by |d_light|
        l = l * (1.0 + d_light) if d_light < 0 else l + (1.0 - l) * d_light
        shifted = colorsys.hls_to_rgb(
            (h + d_hue) % 1.0,
            min(max(l, _WRAP_LIGHTNESS_MIN), _WRAP_LIGHTNESS_MAX),
            min(max(s * s_scale, 0.0), 1.0),
        )
        out.append(_rgb_to_hex(shifted) if as_hex else shifted)
    return out


def _rgb_to_hex(rgb):
    """``(r, g, b)`` floats in 0..1 -> ``#RRGGBB``."""
    return "#{:02X}{:02X}{:02X}".format(
        *(int(round(min(max(c, 0.0), 1.0) * 255)) for c in rgb)
    )
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# NOTE: Palettes for script/visualization/viz_*.py
# ----------------------------------------------------------------
# Shared by viz_tl_responses.py, viz_ad_responses.py and
# viz_detection_responses.py, which each used to carry a private copy of this
# block. Names are kept as the viz scripts already used them (C_*), so those
# scripts import by name. Geometry / font constants stay in the viz scripts.

# ── Response-figure palette (light theme) ──
C_FIG_BG = "#FFFFFF"
C_BOX_EDGE = "#4B5563"  # dark grey
C_SEP = "#D1D5DB"
C_HEADER = "#1D4ED8"
C_TEXT = "#111827"
C_THINK = "#9CA3AF"
C_REASON = "#0284C7"
C_STEP_ANS = "#16A34A"
C_ANS = "#EA580C"
C_TOOL = "#7C3AED"
C_TAG_GREY = "#6B7280"  # display color for all <> tags

# Prompt section highlights
C_IMG_PROMPT = "#D97706"
C_LABEL_NAME = "#059669"  # green-600 — label / landmark / line description
C_IMG_SIZE = "#1D4ED8"
C_PIXEL_SIZE = "#6D28D9"  # violet-700, distinct from C_TOOL (#7C3AED)

# Measurement results
C_MAJ_LEN = "#0F766E"  # teal-700 — major axis length / A-D measurement
C_MIN_LEN = "#BE185D"  # pink-700 — minor axis length

# GT overlay colors (distinct from each other and from prediction colors)
C_GT_MAJOR = "#A21CAF"  # fuchsia-700 — GT major axis / landmark / line 1
C_GT_MINOR = "#4F46E5"  # indigo-600  — GT minor axis / line 2
C_GT_BOX = "#2ECC71"    # green  — GT bounding box
C_PRED_BOX = "#F37020"  # orange — predicted bounding box

# Response tag colors, as a set (used to test whether a token is a tag)
tag_colors = frozenset({C_THINK, C_REASON, C_STEP_ANS, C_ANS, C_TOOL})

# 8 coordinate colors — chosen to avoid conflicts with all fixed token colors.
# TL: [0:4] = P1/P2 major (x, y), [4:8] = P1/P2 minor (x, y).
# A-D: step 1 uses [0:4], step 2 uses [4:8].
response_coord_colors = [
    "#DC2626",  # red-600
    "#F97316",  # orange-500
    "#EAB308",  # yellow-500
    "#84CC16",  # lime-400
    "#06B6D4",  # cyan-500
    "#0EA5E9",  # sky-500
    "#A21CAF",  # fuchsia-700
    "#4F46E5",  # indigo-600
]

# 4 coordinate colors for bounding boxes — lower-left (x, y), upper-right (x, y).
detection_coord_colors = [
    "#DC2626",  # red-600    — lower-left x
    "#F97316",  # orange-500 — lower-left y
    "#06B6D4",  # cyan-500   — upper-right x
    "#0EA5E9",  # sky-500    — upper-right y
]

# Overlay landmark dot colors (plot_tl_axes_on_image / plot_ad_on_image convention).
landmark_dot_colors = ["#4285F4", "#EA4335", "#FDB813", "#34A853"]

# ── Webpage "Case Viewer" section boxes (from medvision-vlm.github.io index.css) ──
C_SEG_FILL = "#ffffff"
C_SEG_EDGE = "#e6e9ef"
C_RAIL_TEAL = "#0E8C8B"  # Prompt & Metrics left rail + pill
C_RAIL_INDIGO = "#4f46e5"  # Response left rail + pill
C_FIGBOX_FILL = "#ffffff"
C_FIGBOX_EDGE = "#e6e9ef"

# ── viz_radar.py ──
# Per-model series colors; cycled through extend_palette when a figure has more
# models than colors. Hex mirror of matplotlib's tab10.
radar_model_colors = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]
C_TUMOR_LESION_LABEL = "#770087"  # purple — tumor / lesion spoke labels
C_ANATOMY_LABEL = "black"         # all other spoke labels

# ── viz_ellipse_fit_comparison.py ──
IMG_COLOR = "#EA4335"   # image-space fit (red)
REAL_COLOR = "#4285F4"  # real-space fit (blue)
MASK_COLOR = "#2ECC71"  # green — mask contour
# ----------------------------------------------------------------
