# Hosting 3D Medical Image Datasets on Hugging Face: A Deep Dive into MedVision

> [!NOTE]
> *   **MedVision** relies on a remote data loading script.
> *   `trust_remote_code` is no longer supported in `datasets>=4.0.0`. Please install `datasets` with `pip install datasets==3.6.0`.

Hosting large-scale, complex medical datasets requires more than just uploading files; it demands a robust architecture to handle 3D volumes, diverse modalities, and precise annotations. In this post, we explore how the [MedVision dataset](https://huggingface.co/datasets/YongchengYAO/MedVision) leverages Hugging Face's advanced dataset features to manage this complexity.

## 🩻 What is MedVision?

[MedVision](https://github.com/YongchengYAO/MedVision) is a large-scale, multi-anatomy, multi-modality dataset designed for **quantitative medical image analysis**. It standardizes **22 public datasets** (like *BraTS24*, *MSD*, *OAIZIB-CM*) — covering XR, CT, MRI, US, and PET — into a unified structure with **30.8M image-annotation pairs**, suitable for training massive foundation models. Images are stored as 3D volumes in RAS+ orientation, and annotations carry **real-world units** (mm and degrees) derived from the physical spacing in the image headers.

### ✨ Key Features

1.  **📦 Automatic Data Handling**: Automatic downloading and processing of 3D images.
2.  **✂️ Dynamic Slicing**: Dynamic loading of 2D slices from local 3D volumes, along any of the three anatomical planes (axial, coronal, sagittal).
3.  **📏 Quantitative Annotations**: Detailed annotations including bounding boxes, mask size, tumor/lesion size, and angle/distance measurements.
4.  **🏗️ Dataset Codebase**: A dedicated codebase (`medvision_ds`) for robust dataset construction.

> [!TIP]
> For the dataset/data-config naming conventions and the fields returned by `load_dataset()`, see the [Essential Dataset Concept](https://github.com/YongchengYAO/MedVision#-essential-dataset-concept) section of the repository README.

## 🛠️ Deep Dive: The Data Processing Script

The magic behind MedVision's integration with the `datasets` library lies in its processing script, [`MedVision.py`](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/MedVision.py). This script orchestrates everything from dependency management to dynamic data slicing.

### 1. ⚙️ Configuration Definition (`MedVisionConfig`)

Medical datasets often require distinct subsets. For example, a user might need **2D sagittal slices** for segmenting target A, but **2D axial slices** for target B. To handle this, MedVision implements a custom configuration class inheriting from `datasets.BuilderConfig`.

**How it works:**
The `MedVisionConfig` class defines essential parameters like `taskType`, `imageType` (2D/3D), and `imageSliceType` (axial/coronal/sagittal) to ensure the correct data view is loaded. Basically, data configuration defines what data will be extracted from the raw data storage.

> [!TIP]
> **Documentation**: [Hugging Face BuilderConfig](https://huggingface.co/docs/datasets/en/package_reference/builder_classes#datasets.BuilderConfig)

### 2. 🗂️ Data Preparation (`_split_generators`)

The `_split_generators` method is responsible for downloading the data and organizing it into splits (Train/Test). The split is made **at the subject level** (70/30), so slices from one subject never appear in both splits.

**Key Features:**

1.  **🧩 Dataset Codebase**: Usage of `medvision_ds`, a distinct codebase located in the [src](https://huggingface.co/datasets/YongchengYAO/MedVision/tree/main/src) folder of the repo. This handles the heavy lifting of data downloading, processing, and annotation generation (via benchmark planners).
    
    > [!TIP]
    > For advanced usage and installation instructions, see the [official guide](https://huggingface.co/datasets/YongchengYAO/MedVision#advanced-usage).
2.  **📥 Raw Image Download**: It checks the `MedVision_DATA_DIR` environment variable and saves the data there.
3.  **🔄 Preprocessing**: It invokes specific download scripts to fetch raw image files and standardizes them (e.g., converting to NIfTI format, reorienting to RAS+ orientation).
4.  **📝 Annotation Handling**: It loads annotations and metadata directly from the benchmark planners, the JSON files released within the dataset repository.

> [!TIP]
> **Documentation**: [Hugging Face SplitGenerator](https://huggingface.co/docs/datasets/en/package_reference/builder_classes#datasets.SplitGenerator)

### 3. 💾 Data Loading (`_generate_examples`)

This method yields the actual training samples. For 3D medical images, simply reading a 3D volume file isn't always sufficient, as many current Vision-Language Models (VLMs) operate on 2D image inputs. Therefore, a flexible method to load 2D slices from 3D volumes is essential.

**How it works:**
For a specific dataset configuration, the script iterates through the cases in the benchmark planner. It dynamically processes the data and filters out invalid samples.

> [!TIP]
> **Documentation**: [Hugging Face: Build and Load](https://huggingface.co/docs/datasets/en/about_dataset_load#build-and-load)

## 📥 Data Downloading & Advanced Usage

While the script automates much of the process, some datasets (like *SKM-TEA* or *ToothFairy2*) have restrictive licenses that prevent direct automatic downloading. For these, MedVision provides a **Data Downloading** guide. Users must manually download the raw data, process it using the provided tools (`python -m medvision_bm.benchmark.download_datasets`), and format it correctly before the Hugging Face script can load it.

*   **Read more**: [MedVision Data Downloading Guide](https://github.com/YongchengYAO/MedVision#-data-downloading-optional)

***

## 🌟 Key Takeaways for Your Own Datasets

1.  **Use `BuilderConfig`** to organize complex datasets with multiple subsets or tasks.
2.  **Automate Installation** inside `_split_generators` if your dataset requires custom helper code.
3.  **Process Dynamically** in `_generate_examples` to save disk space and allow for flexible data views (e.g., generating 2D slices from 3D volumes on the fly).
