# PrevOccupAI-HAR: A Public Domain Dataset for Smartphone Sensor-based Human Activity Recognition in Office Environments

## 1. Abstract
PrevOccupAI-HAR, presents a new publicly available dataset designed to advance smartphone-based human activity recognition 
(HAR) in office environments. PrevOccupAI-HAR comprises two sub-datasets: (1) a model development dataset collected under 
controlled conditions, featuring 20 subjects performing nine sub-activities associated to three main activity classes 
(sitting, standing, and walking), and (2) a real-world dataset captured in an unconstrained office setting captured 
from 13 subjects carrying out their daily office work for six hours continuously. Three machine learning models, namely 
k-nearest neighbors (KNN), support vector machine (SVM), and random forest, were trained on the model development 
dataset to classify the three main classes independently of sub-activity variation. The models achieved accuracies of 
89.80&nbsp;%, 89.98&nbsp;%, and 92.10&nbsp;% for the KNN, SVM, and Random Forest, respectively, on the development 
dataset. When deployed on the real-world dataset, the models attained mean accuracies of 73.27&nbsp;%, 79.97&nbsp;%, and 
77.20&nbsp;%, reflecting performance degradations between 10.01&nbsp;% and 16.53&nbsp;%. Analysis of sequential 
predictions revealed frequent short-duration misclassifications, predominantly between sitting and standing, resulting 
in unstable model outputs. The findings highlight key challenges in transitioning HAR models from controlled to 
real-world contexts and point to future research directions involving temporal deep learning architectures or 
post-processing methods to enhance prediction consistency.

## 2. Related Publication
The presented code base is part of the journal article "PrevOccupAI-HAR: A Public Domain Dataset for Smartphone 
Sensor-based Human Activity Recognition in Office Environments" submitted to the MDPI special issue "Smart Devices and 
Wearable Sensors: Recent Advances and Prospects". For further details on the methodology, please consider reading the
the publication.

The article can be accessed at: (INSERT LINK ONCE PUBLISHED)

To cite the article please use: (INSERT CITATION ONCE PUBLISHED)

## 3. Dataset

The two sub-datasets used in this repository can be downloaded at: (INSERT LINK ONCE DATASET AVAILABLE)

For both datasets, data was acquired using a Xiamoi Redmi Note 9 smartphone that was strapped to the subject's chest.
<center">
<img src="./figures/phone_placement.svg" alt="Placement of smartphone for data acquisition" width="250"/>
</center>

The purpose of this placement is: 
1. it allows for tracking human activities
2. when the subject is seated the trunk movement can be collected, effectively allowing for characterization of seated postures throughout the 
3. this placement can also be utilized as a proxy for development of sensorised garments.

### 2.1 Model Development Dataset
The Model Development Dataset (MD) consists of 20 healthy subjects, comprising 14 women and 6 men, aged between 19 and 
54 years (27 $\pm$ 3.7 years). The dataset contains nine sub-activities that can be associated to three main activities,
namely sitting, standing, and walking.
