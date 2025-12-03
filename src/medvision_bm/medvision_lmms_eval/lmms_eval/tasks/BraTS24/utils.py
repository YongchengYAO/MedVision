from lmms_eval.tasks.medvision.medvision_utils import (
    aggregate_results_avgMAE,
    aggregate_results_avgMRE,
    aggregate_results_MAE,
    aggregate_results_MRE,
    aggregate_results_SuccessRate,
    create_doc_to_text_BoxCoordinate,
    create_doc_to_text_MaskSize,
    create_doc_to_text_TumorLesionSize,
    create_doc_to_text_TumorLesionSize_CoT,
    doc_to_target_BoxCoordinate,
    doc_to_target_MaskSize,
    doc_to_target_TumorLesionSize,
    doc_to_visual,
    process_results_BoxCoordinate,
    process_results_MaskSize,
    process_results_TumorLesionSize,
)
from medvision_ds.datasets.BraTS24 import (
    preprocess_biometry,
    preprocess_detection,
    preprocess_segmentation,
)

doc_to_text_BoxCoordinate = create_doc_to_text_BoxCoordinate(preprocess_detection)
doc_to_text_TumorLesionSize = create_doc_to_text_TumorLesionSize(preprocess_biometry)
doc_to_text_MaskSize = create_doc_to_text_MaskSize(preprocess_segmentation)

doc_to_text_TumorLesionSize_CoT = create_doc_to_text_TumorLesionSize_CoT(preprocess_biometry)
