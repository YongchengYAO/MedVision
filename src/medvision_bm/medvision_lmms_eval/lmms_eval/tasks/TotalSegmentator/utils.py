from lmms_eval.tasks.medvision.medvision_utils import (
    aggregate_results_avgMAE,
    aggregate_results_avgMRE,
    aggregate_results_MAE,
    aggregate_results_MRE,
    aggregate_results_SuccessRate,
    create_doc_to_text_BoxCoordinate,
    create_doc_to_text_BoxCoordinate_CoT,
    create_doc_to_text_BoxCoordinate_wBox,
    doc_to_text_BoxCoordinate_wBox_woMedImg,
    create_doc_to_text_MaskSize,
    create_doc_to_text_MaskSize_wMask,
    doc_to_text_MaskSize_wMask_woMedImg,
    doc_to_target_BoxCoordinate,
    doc_to_target_MaskSize,
    doc_to_visual,
    doc_to_visual_wBox,
    doc_to_visual_wBox_woMedImg, 
    doc_to_visual_wMask,
    doc_to_visual_wMask_woMedImg,
    process_results_BoxCoordinate,
    process_results_MaskSize,
)
from medvision_ds.datasets.TotalSegmentator import (
    preprocess_detection,
    preprocess_segmentation,
)

doc_to_text_BoxCoordinate = create_doc_to_text_BoxCoordinate(preprocess_detection)
doc_to_text_BoxCoordinate_CoT = create_doc_to_text_BoxCoordinate_CoT(preprocess_detection)
doc_to_text_BoxCoordinate_wBox = create_doc_to_text_BoxCoordinate_wBox(preprocess_detection)

doc_to_text_MaskSize = create_doc_to_text_MaskSize(preprocess_segmentation)
doc_to_text_MaskSize_wMask = create_doc_to_text_MaskSize_wMask(preprocess_segmentation)
