from medvision_ds.datasets.FLARE22 import preprocess_detection, preprocess_segmentation

from lmms_eval.tasks.medvision.medvision_utils import (
    doc_to_visual,
    doc_to_target_BoxCoordinate,
    doc_to_target_MaskSize,
    process_results_BoxCoordinate,
    process_results_MaskSize,
    aggregate_results_MAE,
    aggregate_results_MRE,
    aggregate_results_avgMAE,
    aggregate_results_avgMRE,
    aggregate_results_SuccessRate,
)
from lmms_eval.tasks.medvision.medvision_utils import create_doc_to_text_BoxCoordinate, create_doc_to_text_MaskSize 

doc_to_text_BoxCoordinate = create_doc_to_text_BoxCoordinate(preprocess_detection)
doc_to_text_MaskSize = create_doc_to_text_MaskSize(preprocess_segmentation)
