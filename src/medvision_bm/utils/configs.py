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
SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_METRICS = (
    "summary_metrics_per_sample_detect_Task.csv"
)
SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_MEAN_METRICS = (
    "summary_metrics_boxImgRatio_x_label_detect_Task.csv"
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
    "AMOS22": "AMOS22",
    "AbdomenAtlas1.0Mini": "AbdomenAtlas__1_0__Mini",
    "AbdomenCT-1K": "AbdomenCT_1K",
    "BCV15": "BCV15",
    "BraTS24": "BraTS24",
    "CAMUS": "CAMUS",
    "Ceph-Biometrics-400": "Ceph_Biometrics_400",
    "CrossMoDA": "CrossMoDA",
    "FLARE22": "FLARE22",
    "FeTA24": "FeTA24",
    "HNTSMRG24": "HNTSMRG24",
    "ISLES24": "ISLES24",
    "KiPA22": "KiPA22",
    "KiTS23": "KiTS23",
    "MSD": "MSD",
    "OAIZIB-CM": "OAIZIB_CM",
    "SKM-TEA": "SKM_TEA",
    "ToothFairy2": "ToothFairy2",
    "TopCoW24": "TopCoW24",
    "TotalSegmentator": "TotalSegmentator",
    "autoPET-III": "autoPET_III",
}
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
    # ───────────────────────────── THROAT & AIRWAY ───────────────────
    # head & neck
    "cochlea": "Head-Neck",
    "trachea": "Head-Neck",
    "pharynx": "Head-Neck",
    "thyroid gland": "Head-Neck",
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
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "S1",
        )
    },
    "vertebrae": "Spine",
    "intervertebral discs": "Spine",
    "spinal cord": "Spine",
    # knee
    "femur": "Knee",
    "tibia": "Knee",
    "left femur": "Knee",
    "right femur": "Knee",
    "femoral cartilage": "Knee",
    "lateral tibial cartilage": "Knee",
    "medial tibial cartilage": "Knee",
    "patellar cartilage": "Knee",
    "lateral meniscus": "Knee",
    "medial meniscus": "Knee",
    # lymphatics
    "metastatic lymph node": "Metastatic Lymph Node",
    # miscellaneous pathology (non-organ specific)
    "edema": "Miscellaneous Tumor/Lesion",
    "tumor": "Miscellaneous Tumor/Lesion",
    "cystic component": "Miscellaneous Tumor/Lesion",
    # dentistry
    "upper jawbone": "Jawbone",
    "lower jawbone": "Jawbone",
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
