# BE 645 Artificial Intelligence (AI) and Radiomics (Spring 2025) - Updated

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

**Full Playlists**:

> Recent Playlist from Spring 2025 (AI-Generated
> Podcasts): https://www.youtube.com/playlist?list=PLVrN2LRb7eT0VBZqrtSAJQd2mqVtIDJKx

> Old Playlist from Summer 2024 (Recorded): https://www.youtube.com/playlist?list=PLVrN2LRb7eT2KV3YMdXeF2B9dgaN4QF4g

**AI-Generated Podcasts (Spring 2025)**:

1. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 01 Recap (AI-Generated Podcast)](https://youtu.be/E1NFmke8FCs)
2. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 02 Recap (AI-Generated Podcast)](https://youtu.be/TiBbfKyX9AI)
3. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 03 Recap (AI-Generated Podcast)](https://youtu.be/Tl-cWlO5p8E)
4. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 04 Recap (AI-Generated Podcast)](https://youtu.be/N5vN7UihUIs)
5. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 05 Recap (AI-Generated Podcast)](https://youtu.be/-Ke9wfCHYgs)
6. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 06 Recap (AI-Generated Podcast)](https://youtu.be/ykwc6M0TJ4M)
7. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 07 Recap (AI-Generated Podcast)](https://youtu.be/5NLd2JnyZYc)
8. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 08 Recap (AI-Generated Podcast)](https://youtu.be/MnhxJRPTyV4)

**Videos (Summer 2024)**:

1. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 01 - Introduction](https://youtu.be/pefwr1HP_wA)
2. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 02 - First Order Features](https://youtu.be/MpCFet8SEC4)
3. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 03 - GLCM](https://youtu.be/wrvbEjg3bw8)
4. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 04 - GLRLM](https://youtu.be/0z39SjZMuyI)
5. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 05 - GLSZM](https://youtu.be/djnwNE1_8cI)
6. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 06 - Machine Learning Example](https://youtu.be/0Xk4eztvMHc)
7. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 07 - LBP](https://youtu.be/ZKJpComTCoQ)
8. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 08 - Shape Features](https://youtu.be/OO4KKR8KkJ4)
9. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 09 - Machine Learning 3D Example](https://youtu.be/EXaAj1syKvk)
10. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 10 - Machine Learning 3D Example Part-2](https://youtu.be/bFPzK4QFPQQ)
11. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 11 - Building & Tuning ML-based System](https://youtu.be/rFFF3PhxMt4)

## Programming Language and Libraries

The programming language used in this series is `Python`, and the primary libraries employed are:

1. `OpenCV` - An open-source computer vision and machine learning software library.
2. `NumPy` - A fundamental package for scientific computing with Python.
3. `Matplotlib` - A comprehensive library for creating static, animated, and interactive visualizations in Python.
4. `Scikit-learn` - A simple and efficient tool for data mining and data analysis built on NumPy, SciPy, and Matplotlib.
5. `Split Folders` - A simple library to split folders into training, validation, and testing directories.
6. `NiBabel` - A library to read and write common neuroimaging file formats.
7. `Pandas` - A fast, powerful, flexible, and easy-to-use open-source data analysis and data manipulation library.
8. `TQDM` - A fast, extensible progress bar for loops and CLI.
10. `Trimesh` - A pure library for loading and using triangular meshes with an emphasis on watertight meshes.
11. `pyglet` - A cross-platform windowing and multimedia library for Python.
12. `Scikit-image` - A collection of algorithms for image processing.
13. `Imbalanced-learn` - A Python library to tackle the problem of imbalanced datasets.
14. `XGBoost` - An optimized distributed gradient boosting library designed to be highly efficient, flexible, and
    portable.
15. `Optuna` - An automatic hyperparameter optimization software framework that is suitable for machine learning tasks.

The packages versions of the libraries used in this series are:

```text
cv2==4.9.0
numpy==1.26.4
matplotlib==3.8.3
scikit-learn==1.4.1.post1
splitfolders==0.5.1
nibabel==5.2.1
pandas==2.2.1
tqdm==4.66.2
trimesh==4.4.1
pyglet==1.5.29
scikit-image==0.22.0
imbalanced-learn=0.12.3
xgboost=2.1.0
optuna=3.6.1
```

To install the required libraries, you can use the following PIP commands:

```
pip install opencv-python==4.9.*
pip install opencv-contrib-python==4.9.*
pip install numpy==1.26.4
pip install matplotlib==3.8.3
pip install scikit-learn==1.4.1.post1
pip install split-folders==0.5.1
pip install nibabel==5.2.1
pip install pandas==2.2.1
pip install tqdm==4.66.2
pip install trimesh==4.4.1
pip install pyglet==1.5.29
pip install scikit-image==0.22.0
pip install imbalanced-learn==0.12.3
pip install xgboost==2.1.0
pip install optuna==3.6.1
```

_Disclaimer: The versions of the libraries may change based on updates and releases. However, the code should work
with the latest versions. Please note that the code has been tested on `Python 3.9.17` and the specified library
versions on a `Windows 11` machine. It has not been tested on other operating systems or other versions of Python and
the libraries._

## Dataset and Code

**Dataset**:

***Liver Tumor Segmentation (130 CT Scans for Liver Tumor Segmentation)***

This dataset was extracted from LiTS - Liver Tumor Segmentation Challenge (LiTS17) organised in conjunction with ISBI
2017 and MICCAI 2017.

Dataset Link: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation

More information: Original dataset is The Liver Tumor Segmentation Benchmark (LiTS) that can be accessed from this
link: https://arxiv.org/abs/1901.04056

Citation for the dataset:

> Bilic, P., Christ, P., Li, H. B., Vorontsov, E., Ben-Cohen, A., Kaissis, G., ... & Menze, B. (2023). The liver tumor
> segmentation benchmark (lits). Medical Image Analysis, 84, 102680.

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

Citation for the database:

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
copyright law.
For permission requests, contact the author.

The code provided in this series is for educational purposes only and should be used with caution.
The author is not responsible for any misuse of the code provided.

If you need to use the code for research or commercial purposes, please contact the author for a written permission.

## Citations and Acknowledgments

If you find this series helpful and use it in your research or projects, please consider citing it as:

```bibtex
@software{Balaha_BE_645_Artificial_2024,
  author  = {Balaha, Hossam Magdy},
  month   = jun,
  title   = {{BE 645 Artificial Intelligence (AI) and Radiomics (Summer 2024)}},
  url     = {https://github.com/HossamBalaha/BE-645-Artificial-Intelligence-and-Radiomics},
  version = {1.06.19},
  year    = {2024}
}

@software{hossam_magdy_balaha_2024_12170422,
  author    = {Hossam Magdy Balaha},
  title     = {{HossamBalaha/BE-645-Artificial-Intelligence-and-Radiomics: v1.06.19}},
  month     = jun,
  year      = 2024,
  publisher = {Zenodo},
  version   = {v1.06.19},
  doi       = {10.5281/zenodo.12170422},
  url       = {https://doi.org/10.5281/zenodo.12170422}
}
```

## Contact

This series is prepared and presented by `Hossam Magdy Balaha` from the University of Louisville's J.B. Speed School of
Engineering.

For any questions or inquiries, please contact me using the contact information available on my CV at the following
link: https://hossambalaha.github.io/
