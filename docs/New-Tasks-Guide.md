# Adding New Tasks to MedVision

**Concept:** Adding new tasks means defining datasets, mapping functions that convert data into prompts and targets, post-processing functions for parsing outputs, and metric calculation functions.

## Codebase Architecture

```
├── src
	├── medvision_bm 
		├── medvision_lmms_eval
			├── lmms_eval
				├── tasks # [1]
					├── BraTS24 # [2]
						├── utils.py # [3]
						├── BraTS24_BoxCoordinate_base-CoT.yaml # [4]
						├── BraTS24_BoxCoordinate_Task01_Axial-CoT.yaml # [5]
						├── BraTS24_TumorLesionSize_base-CoT.yaml # [6]
						├── BraTS24_TumorLesionSize_Task01_Axial-CoT.yaml # [7]
						├── BraTS24_MaskSize_base.yaml # [8]
						├── BraTS24_MaskSize_Task01_Axial.yaml # [9]
					├── Ceph-Biometrics-400
						├── Ceph-Biometrics-400_BiometricsFromLandmarks_base-CoT.yaml # [10]
						├── Ceph-Biometrics-400_BiometricsFromLandmarks_Angle_Task01_Sagittal-CoT.yaml # [11]
					├── <other datasets>
					├── medvision 
						├── lmms_eval_specific_kwargs.yaml
						├── medvision_utils.py # [12]
						├── metadata.yaml  
```

- [1] Tasks folder

- [2] Dataset folder (example: BraTS24)

- [3] Utility functions shared for all tasks of the same dataset 

- [4,6,8,10] Base task configuration file

  e.g., `BraTS24_BoxCoordinate_base-CoT.yaml`:

  ```yaml
  include:
  - ../medvision/lmms_eval_specific_kwargs.yaml
  - ../medvision/metadata.yaml
  tag: MedVision-BoxCoordinate,BraTS24
  dataset_path: YongchengYAO/MedVision
  dataset_kwargs:
    token: True
    trust_remote_code: True
    # download_mode: force_redownload
  test_split: test
  fewshot_split: test
  num_fewshot: 0
  output_type: generate_until
  doc_to_visual: !function utils.doc_to_visual
  doc_to_text: !function utils.doc_to_text_BoxCoordinate_CoT
  doc_to_target: !function utils.doc_to_target_BoxCoordinate
  process_results: !function utils.process_results_BoxCoordinate
  metric_list:
  - metric: avgMAE
    aggregation: !function utils.aggregate_results_avgMAE
    higher_is_better: false
  - metric: avgMRE
    aggregation: !function utils.aggregate_results_avgMRE
    higher_is_better: false
  - metric: SuccessRate
    aggregation: !function utils.aggregate_results_SuccessRate
    higher_is_better: true
  ```

  - `dataset_path`: HF dataset ID
  - these functions are defined in `utils.py` 
    - `doc_to_visual`:  get the image input
    - `doc_to_text`: construct text prompt
    - `doc_to_target`: construct target string
    - `process_results`: compute per-sample metrics
    - `aggregation`:  compute average metrics

- [5,7,9,11] Task definition: set task and dataset names

  e.g., `BraTS24_BoxCoordinate_Task01_Axial-CoT.yaml`:

  ```yaml
  include: BraTS24_BoxCoordinate_base-CoT.yaml
  task: BraTS24_BoxCoordinate_Task01_Axial-CoT
  dataset_name: BraTS24_BoxSize_Task01_Axial_Test
  ```

- [12] A central place for all utilities

## Keys

1. Keep the naming conventions:

   - task template should be `<dataset>_<task_type>_base.yaml`
   - task should be `<dataset>_<task_type>_<task_num>_<slice><other_tag>.yaml`
   - Use fixed `<task_type>` labels:
     - `BoxCoordinate` for detection tasks
     - `TumorLesionSize` for tumor/lesion size estimation tasks
     - `BiometricsFromLandmarks` for angle/distance measurement tasks
     - **[New/Preview]** `MaskSize` for area estimation tasks

2. **Add new tasks** in a new dataset folder and folllow the file structure:

   ```
   ├── src
   	├── medvision_bm 
   		├── medvision_lmms_eval
   			├── lmms_eval
   				├── tasks
   					├── <new-dataset>
   						├── utils.py
   						├── <dataset>_<task_type>_base.yaml
   						├── <dataset>_<task_type>_<task_num>_<slice><other_tag>.yaml
   ```

3. Add new functions to `medvision_utils.py ` and import in  `utils.py`

## New Tasks on New Datasets

If you wish to add new tasks on new datasets, refer to the [New Datasets Guide](https://huggingface.co/datasets/YongchengYAO/MedVision#new-datasets-guide) from HF dataset `YongchengYAO/MedVision`.


## Reference

- If you want to build a HF dataset like MedVision, read this [blog](https://huggingface.co/blog/YongchengYAO/medvision-dataset).

