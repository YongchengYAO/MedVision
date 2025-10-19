from medvision_ds.datasets.MSD import preprocess_detection, preprocess_biometry, preprocess_segmentation

from lmms_eval.tasks.medvision.medvision_utils import (
    doc_to_visual,
    doc_to_target_BoxCoordinate,
    doc_to_target_MaskSize,
    doc_to_target_TumorLesionSize,
    process_results_BoxCoordinate,
    process_results_MaskSize,
    process_results_TumorLesionSize,
    aggregate_results_MAE,
    aggregate_results_MRE,
    aggregate_results_avgMAE,
    aggregate_results_avgMRE,
    aggregate_results_SuccessRate,
)
from lmms_eval.tasks.medvision.medvision_utils import create_doc_to_text_BoxCoordinate, create_doc_to_text_TumorLesionSize, create_doc_to_text_MaskSize

doc_to_text_BoxCoordinate = create_doc_to_text_BoxCoordinate(preprocess_detection)
doc_to_text_TumorLesionSize = create_doc_to_text_TumorLesionSize(preprocess_biometry)
doc_to_text_MaskSize = create_doc_to_text_MaskSize(preprocess_segmentation)
