# BE 645 Artificial Intelligence (AI) and Radiomics (Spring 2026) - Updated

Welcome to the BE 645: Artificial Intelligence (AI) and Radiomics course.

Artificial intelligence is essentially a collection of advanced computational algorithms designed to identify patterns
in given data and predict outcomes for new, unseen data. Radiomics, a relatively new term in radiology, involves
extracting a large number of features from various types of medical images. This course integrates artificial
intelligence and radiomics to uncover valuable quantitative data for practical medical applications. It also covers the
fundamental concepts and applications of artificial intelligence in computer-aided diagnostic systems.

This course offers both theoretical and practical knowledge about computer vision and AI techniques essential for
processing and analyzing radiology images, contributing to the shift towards radiomics. This transition will
allow AI models to assist doctors and healthcare professionals in managing and diagnosing various diseases.

> If you encountered any issues or errors in the code or lectures, please feel free to let me know. I will be more than
> happy to fix them and update the repository accordingly. Your feedback is highly appreciated and will help me improve
> the quality of the content provided in this series.

## Full Playlist and Videos

This series is your gateway to the fascinating world of applying AI techniques to radiomics.

**Earlier Playlists**:

> Playlist from Summer 2025 (Recorded): https://www.youtube.com/playlist?list=PLVrN2LRb7eT2GOJS8YKf1TcP6X1jr-9Dn

> Playlist from Spring 2025 (AI-Generated Podcasts):
> https://www.youtube.com/playlist?list=PLVrN2LRb7eT0VBZqrtSAJQd2mqVtIDJKx

> Playlist from Summer 2024 (Recorded): https://www.youtube.com/playlist?list=PLVrN2LRb7eT2KV3YMdXeF2B9dgaN4QF4g

## Programming Language and Libraries

This project is written in Python. All Python package dependencies required by the lectures and examples are listed in
the `requirements.txt` file at the repository root.

You can install the dependencies directly with pip (system / virtualenv / activated conda env):

```cmd
pip install -r requirements.txt
```

Recommended Python version: Python `3.10` (the materials were developed and tested with Python `3.10.x`, e.g. `3.10.18`)
on a Windows machine. The code will often work with other Python `3.10.x` builds, but behavior on other Python
major/minor versions or on other operating systems has not been exhaustively tested.

## Anaconda Environment Setup (Optional But Recommended)

A helper batch script is provided to automate creating a Conda environment and installing the packages from
`requirements.txt` on Windows:

Script: `anaconda-tf-environment.bat` (located in the repository root)

Key points about what the script does and how it behaves:

- Defaults: environment name `be645` and Python `3.10` (you may supply a different name and Python version as positional
  arguments).
- Supported flags: `--no-gpu` (skip attempting to install CUDA/cuDNN), `--force` (remove any existing environment with
  the same name before creating), `--no-pause` (do not pause at the end), `--silent` (suppress console output), and
  `--help`.
    - The script also accepts `--quiet` as an alias for `--silent`.
- The script verifies that `conda` is available on PATH; if not, it exits with a message and non-zero status. Run it
  from an Anaconda Prompt or enable conda in your shell before using the script.
- It attempts a non-fatal `conda update -n base -c defaults conda` early on; the script continues even if the update
  fails.
- Ensures `requirements.txt` exists next to the script; if missing the script exits with an explanatory error.
- Environment creation and removal are handled via `conda create` / `conda env remove`. Subsequent Python/package
  commands are executed inside the environment using `conda run -n "<env>"` so `conda init` is not required.
- If an NVIDIA GPU is detected (presence of `nvidia-smi` on PATH) and `--no-gpu` is not provided, the script attempts to
  install `cudatoolkit` and `cudnn` into the new environment from `conda-forge`.
- The script upgrades `pip` inside the created environment and installs the packages from `requirements.txt` using a
  single pip invocation:
  `conda run -n "<env>" python -m pip install --progress-bar off -r "requirements.txt"`.
- Logging: the script writes a log file named `anaconda-tf-environment.log` next to the script and appends sanitized
  runtime messages. By default the script streams messages to the console and appends the same text to the log; use
  `--silent` to suppress console output while still writing the log.
- By default the script pauses at the end so you can read messages; use `--no-pause` for non-interactive or automated
  runs.

Usage examples (Windows `cmd.exe`):

- Create the default environment named `be645` with Python 3.10 (interactive):

```cmd
anaconda-tf-environment.bat
```

- Create a custom environment `myenv` with Python 3.10 (console shows progress):

```cmd
anaconda-tf-environment.bat myenv 3.10
```

- Create `myenv` but skip GPU/CUDA installation:

```cmd
anaconda-tf-environment.bat myenv 3.10 --no-gpu
```

- Recreate an existing environment (force remove then create):

```cmd
anaconda-tf-environment.bat myenv 3.10 --force
```

- Run non-interactively (do not pause at the end):

```cmd
anaconda-tf-environment.bat myenv 3.10 --no-pause
```

- Run silently (suppress console output, log still written):

```cmd
anaconda-tf-environment.bat --silent
```

If you prefer to set up the environment manually, you can run the following commands in an Anaconda Prompt:

```cmd
conda create -n be645 python=3.10 -y
conda activate be645
pip install -r requirements.txt
```

## Datasets and Code

**Datasets**

***Liver Tumor Segmentation (130 CT Scans for Liver Tumor Segmentation)***

This dataset was extracted from LiTS - Liver Tumor Segmentation Challenge (LiTS17) organized in conjunction with ISBI
2017 and MICCAI 2017.

Dataset Link: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation

More information: Original dataset is The Liver Tumor Segmentation Benchmark (LiTS) that can be accessed from this
link: https://arxiv.org/abs/1901.04056

Citation for the dataset:

> Bilic, P., Christ, P., Li, H. B., Vorontsov, E., Ben-Cohen, A., Kaissis, G., ... & Menze, B. (2023). The liver tumor
> segmentation benchmark (lits). Medical Image Analysis, 84, 102680.

***Brain Tumor Dataset***

This dataset contains 3,064 T1-weighted contrast-enhanced MRI images from 233 patients, categorized into three
brain tumor types: meningioma (708 slices), glioma (1,426 slices), and pituitary tumor (930 slices).
The dataset is divided into four subsets, each archived in a `.zip` file containing 766 slices,
along with 5-fold cross-validation indices for robust evaluation.

Each image is stored in MATLAB (`.mat`) format, including fields such as tumor label, patient ID, image data,
tumor border coordinates, and a binary tumor mask.
The images were acquired using a standardized protocol at Nanfang Hospital and General Hospital,
Tianjin Medical University, China, between 2005 and 2010, with a resolution of 512 x 512 pixels and pixel
dimensions of 0.49 x 0.49 mm².

Dataset Link: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

Citations for the dataset:

> Cheng, J., Huang, W., Cao, S., Yang, R., Yang, W., Yun, Z., ... & Feng, Q. (2015). Enhanced performance of brain tumor
> classification via tumor region augmentation and partition. PloS one, 10(10), e0140381.

> Cheng, J., Yang, W., Huang, M., Huang, W., Jiang, J., Zhou, Y., ... & Chen, W. (2016). Retrieval of brain tumors by
> adaptive spatial pooling and fisher vector representation. PloS one, 11(6), e0157112.

***Brain Tumor Dataset: Segmentation & Classification***

This dataset is a curated and enhanced collection of brain tumor MRI
images derived from two publicly available datasets: the Kaggle Brain Tumor MRI Dataset and the SciDB Brain Tumor
Dataset. It is designed for both segmentation and classification tasks, including identifying tumor types such
as glioma, meningioma, and pituitary tumors, with approximately 5,000 images and ~2,700 segmentation masks. Enhancements
include normalization, noise reduction through Gaussian filtering, contrast adjustments, and structured organization
into directories for images and masks, along with pixel-level annotations and classification labels.

Dataset Link: https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification

***Breast Ultrasound Images Dataset***

The dataset comprises 780 breast ultrasound images collected in 2018 from 600 female patients aged 25 to 75 years.
Each image has an average size of 500 x 500 pixels and is stored in PNG format, accompanied by ground truth images for
reference.
The dataset is structured into three categories: normal, benign, and malignant.

Dataset Link: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

More information: Original paper (Dataset of breast ultrasound images) that can be accessed from this
link: https://doi.org/10.1016/j.dib.2019.104863

Citation for the dataset:

> Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images.
> Data in brief, 28, 104863.

***COVID-19 Radiography Database***

A team of researchers from Qatar University, Doha, the University of Dhaka, and their collaborators from Pakistan and
Malaysia, in collaboration with medical doctors, developed a comprehensive COVID-19 Radiography Database, which won the
COVID-19 Dataset Award by the Kaggle Community.

Initially, they released a dataset containing 219 COVID-19, 1341 normal, and 1,345 viral pneumonia chest X-ray images.
Subsequent updates expanded the dataset significantly, with the latest update including 3,616 COVID-19 cases,
10,192 normal, 6,012 lung opacity, and 1,345 viral pneumonia images, along with corresponding lung masks.

Additionally, Qatar University researchers compiled the COVID-QU-Ex dataset, featuring 33,920 chest X-ray images
and ground-truth lung segmentation masks, making it the largest lung mask dataset created.
The dataset includes 11,956 COVID-19, 11,263 non-COVID infections, and 10,701 normal images,
with 2,913 COVID-19 infection segmentation masks provided from their previous QaTaCov project.

Dataset Link: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Citation for the dataset:

> Chowdhury, M. E., Rahman, T., Khandakar, A., Mazhar, R., Kadir, M. A., Mahbub, Z. B., ... & Islam, M. T. (2020). Can
> AI help in screening viral and COVID-19 pneumonia?. Ieee Access, 8, 132665-132676.

***[MedMNIST+] 18x Standardized Datasets for 2D and 3D Biomedical Image Classification with Multiple Size Options: 28 (
MNIST-Like), 64, 128, and 224***

MedMNIST is a comprehensive collection of standardized biomedical images, designed to simplify research
and educational activities in biomedical image analysis, computer vision, and machine learning.
It includes 12 datasets for 2D images and 6 datasets for 3D images, all pre-processed to 28x28 (2D) or
28x28x28 (3D) with classification labels, eliminating the need for background knowledge.
The dataset spans various data scales (100 to 100,000) and tasks (binary/multi-class classification,
ordinal regression, multi-label classification), totaling around 708K 2D and 10K 3D images.

Recently, MedMNIST+ was released, offering larger image sizes (64x64, 128x128, 224x224 for 2D, and 64x64x64 for 3D)
to support the development of medical foundation models.

Dataset Link: https://zenodo.org/records/10519652

Citations for the dataset:

> Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., ... & Ni, B. (2023). Medmnist v2-a large-scale lightweight
> benchmark for 2d and 3d biomedical image classification. Scientific Data, 10(1), 41.
>
> Yang, J., Shi, R., & Ni, B. (2021, April). Medmnist classification decathlon: A lightweight automl benchmark for
> medical image analysis. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI) (pp. 191-195). IEEE.

_Disclaimer: The datasets are provided for educational purposes only. They are publicly available and can be
accessed from their original links. The author, myself, does not own the datasets._

**Code**:

All code used in the lectures will be available in this GitHub
repository (https://github.com/HossamBalaha/BE-645-Artificial-Intelligence-and-Radiomics) in
the `Lectures Scripts` folder.

## Copyright and License

No part of this series may be reproduced, distributed, or transmitted in any form or by any means, including
photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the author,
except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by
copyright law. For permission requests, contact the author.

The code provided in this series is for educational purposes only and should be used with caution.
The author is not responsible for any misuse of the code provided.

If you need to use the code for research or commercial purposes, please contact the author for a written permission.

## Citations and Acknowledgments

If you find this series helpful and use it in your research or projects, please consider citing it as:

```bibtex
@software{Balaha_BE_645_Artificial_2026,
  author  = {Balaha, Hossam Magdy},
  month   = jan,
  title   = {{BE 645 Artificial Intelligence (AI) and Radiomics}},
  url     = {https://github.com/HossamBalaha/BE-645-Artificial-Intelligence-and-Radiomics},
  version = {1.26.1},
  year    = {2024}
}

@software{hossam_magdy_balaha_2024_12170422,
  author    = {Hossam Magdy Balaha},
  title     = {{HossamBalaha/BE-645-Artificial-Intelligence-and-Radiomics}},
  month     = jan,
  year      = 2024,
  publisher = {Zenodo},
  version   = {v1.26.1},
  doi       = {https://doi.org/10.5281/zenodo.18179368},
  url       = {https://doi.org/https://doi.org/10.5281/zenodo.18179368}
}
```

## Contact

This series is prepared and presented by `Hossam Magdy Balaha` from the University of Louisville's J.B. Speed School of
Engineering.

For any questions or inquiries, please contact me using the contact information available on my CV at the following
link: https://hossambalaha.github.io/
