from medvision_ds.datasets.Ceph_Biometrics_400 import preprocess_biometry 

from lmms_eval.tasks.medvision.medvision_utils import (
    doc_to_visual,
    doc_to_target_BiometricsFromLandmarks,
    process_results_BiometricsFromLandmarks,
    aggregate_results_MAE,
    aggregate_results_MRE,
    aggregate_results_avgMAE,
    aggregate_results_avgMRE,
    aggregate_results_SuccessRate,
)
from lmms_eval.tasks.medvision.medvision_utils import create_doc_to_text_BiometricsFromLandmarks

doc_to_text_BiometricsFromLandmarks = create_doc_to_text_BiometricsFromLandmarks(preprocess_biometry)
