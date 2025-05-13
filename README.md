# Segmentation_FL_Arbeit

## Abstract 

Working with federated learning setups has gained vital importance recently due to concerns over data privacy. Semantic segmentation is a crucial task being used in numerous applications, such as medical imaging, self driving cars, and robotic vision, requires precision and efficient model training. In this project, we delve into implementation of semantic segmentation task within a federated learning framework, focusing deployment on edge devices with limited computational resources. Utilizing the cityscapes dataset, performed segmentation tasks employing lightweight, pre-trained model optimized for resource-constrained environments. This study evaluates three federated learning algorithms, such as Federated Averaging (FedAvg), Federated Dynamic (FedDyn), Federated Optimization in Heterogeneous Networks (FedProx), comparing their performance against traditional centralized training approach. To address the issue of generalizability, we integrated the Minimizing Activation Norm (MAN) regularizer, previously proven effective for classification task, into our segmentation models to assess the impact on performance. This regularizer offers improvement in performance of various
federated learning algorithms like FedAvg and FedDyn in both uniform and non-uniform number of samples distribution, but fails to improve performance in non-uniform samples scenario for FedProx.
This study highlights the potential of federated learning for edge device deployment and the benefits of regularization technique (MAN) in improving model performance


# 📂 Dataset

The dataset used for this project is Open-source cityscapes dataset, which provides high-quality pixel level annotations for urban scene understanding. Dataset can be downloaded from [here](https://www.cityscapes-dataset.com/dataset-overview/)


# 🧠 MAN regularizer

The Minimizing Activations Norm (MAN) regularizer technique was originally proposed for classification tasks, in paper (https://openaccess.thecvf.com/content/WACV2024/papers/Yashwanth_Minimizing_Layerwise_Activation_Norm_Improves_Generalization_in_Federated_Learning_WACV_2024_paper.pdf). Refer to this paper, to know more. In this project, MAN regularizer has been extended and evaluated in context of semantic segmentation tasks to investigate its generalization benefits in a federated learning setup. 


## Federated learning algorithms

In this project, three FL algorithms are tested: 
- FedAvg
- FedDyn
- FedProx.

These algorithms were assessed under 2 types of sampling: uniform (IID) and non-uniform (non-IID) data distributions. Additionally, experiments were conducted with and without the MAN regularizer to demonstrate its impact on generalization performance in federated setups..


## 📊 Report
For detailed experimental results, evaluations, and visualizations, the full report is available [here](



## Sample Result

![segmentation_visualization (Fedavg)](https://github.com/user-attachments/assets/08f3e3e9-a630-46e7-a7ed-34122f325c56)


## 

# Acknowledgment

This project is done under Chair of Computer Graphics and Multimedia Systems Group at Universität Siegen as part of Arbeit course in my Masters
