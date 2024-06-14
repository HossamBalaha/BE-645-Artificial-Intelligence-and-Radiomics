# BE 645 Artificial Intelligence (AI) and Radiomics (Summer 2024)

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

**Full Playlist**:
Link: https://www.youtube.com/playlist?list=PLVrN2LRb7eT2KV3YMdXeF2B9dgaN4QF4g

**Videos**:

1. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 01 - Introduction](https://youtu.be/pefwr1HP_wA)
2. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 02 - First Order Features](https://youtu.be/MpCFet8SEC4)
3. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 03 - GLCM](https://youtu.be/wrvbEjg3bw8)
4. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 04 - GLRLM](https://youtu.be/0z39SjZMuyI)
5. [BE 645: Artificial Intelligence (AI) and Radiomics - Lecture 04 - GLSZM]()

... and more to come!

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
```

_Disclaimer: The versions of the libraries may change based on updates and releases. However, the code should work
with the latest versions. Please note that the code has been tested on `Python 3.9.17` and the specified library
versions on a `Windows 11` machine. It has not been tested on other operating systems or other versions of Python and
the libraries._

## Dataset and Code

**Dataset**:

***Liver Tumor Segmentation (130 CT Scans for Liver Tumor Segmentation)***

This dataset was extracted from LiTS – Liver Tumor Segmentation Challenge (LiTS17) organised in conjunction with ISBI
2017 and MICCAI 2017.

Dataset Link: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation

More information: Original dataset is The Liver Tumor Segmentation Benchmark (LiTS) that can be accessed from this
link: https://arxiv.org/abs/1901.04056

***COVID-19 Radiography Database***

A team of researchers from Qatar University, Doha, the University of Dhaka, and their collaborators from Pakistan and
Malaysia, in collaboration with medical doctors, developed a comprehensive COVID-19 Radiography Database, which won the
COVID-19 Dataset Award by the Kaggle Community. Initially, they released a dataset containing 219 COVID-19, 1341 normal,
and 1,345 viral pneumonia chest X-ray images. Subsequent updates expanded the dataset significantly, with the latest
update including 3,616 COVID-19 cases, 10,192 normal, 6,012 lung opacity, and 1,345 viral pneumonia images, along with
corresponding lung masks. This database will be continually updated as new data becomes available. Additionally, Qatar
University researchers compiled the COVID-QU-Ex dataset, featuring 33,920 chest X-ray images and ground-truth lung
segmentation masks, making it the largest lung mask dataset created. The dataset includes 11,956 COVID-19, 11,263
non-COVID infections, and 10,701 normal images, with 2913 COVID-19 infection segmentation masks provided from their
previous QaTaCov project. The dataset is available on Kaggle for download.

Dataset Link: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Citation for the COVID-19 Radiography Database:

> Chowdhury, M. E., Rahman, T., Khandakar, A., Mazhar, R., Kadir, M. A., Mahbub, Z. B., ... & Islam, M. T. (2020). Can
> AI help in screening viral and COVID-19 pneumonia?. Ieee Access, 8, 132665-132676.

_Disclaimer: The datasets are provided for educational purposes only. They are publicly available and can be
accessed from their original links. The author, myself, does not own the datasets._

**Code**:

All code used in the lectures will be available in this GitHub
repository (https://github.com/HossamBalaha/BE-645-Artificial-Intelligence-and-Radiomics) in
the `Lectures Scripts` folder.

## Contact

This series is prepared and presented by `Hossam Magdy Balaha` from the University of Louisville's J.B. Speed School of
Engineering.

For any questions or inquiries, please contact me using the contact information available on my CV at the following
link: https://hossambalaha.github.io/
