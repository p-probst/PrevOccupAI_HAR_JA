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

For both datasets, data was acquired using a Xiamoi Redmi Note 9 smartphone that was strapped to the subject's chest, 
as shown below.
<p align="center">
<img src="./figures/phone_placement.svg" alt="Placement of smartphone for data acquisition" width="250"/>
</p>

The purpose of this placement is: 
1. it allows for tracking human activities.
2. when the subject is seated the trunk movement can be collected, effectively allowing for characterization of seated postures throughout the work day. 
3. this placement can also be utilized as a proxy for development of sensorised garments.

The utilized smartphone runs on the Android operating system (OS) and was used to acquire tri-axial accelerometer (ACC), 
gyroscope (GYR), and magnetometer (MAG) data, as well as rotation vector (ROT) data. ACC, GYR, and ROT were acquired at 
`100 Hz`, while MAG was sampled at `50 Hz` due to restrictions of Android OS.

A total of 27 subjects participated in the data collection, some of which participated in both datasets. The subject 
details for both datasets, comprising subject ID, sex, age, and participation in which of the datasets, are shown in 
table below.

| Subject ID | Sex | Age (years) | Model development | Model evaluation |
|------------|-----|-------------|-------------------|------------------|
| P001 | F | 22 | ✓ | ✓ |
| P002 | F | 22 | ✓ | ✓ |
| P003 | F | 22 | ✓ | — |
| P004 | F | 54 | ✓ | — |
| P005 | M | 22 | ✓ | ✓ |
| P006 | F | 23 | ✓ | — |
| P007 | F | 20 | ✓ | — |
| P008 | M | 33 | ✓ | ✓ |
| P009 | M | 23 | ✓ | — |
| P010 | F | 22 | ✓ | — |
| P011 | F | 24 | ✓ | — |
| P012 | M | 24 | ✓ | ✓ |
| P013 | F | 21 | ✓ | — |
| P014 | F | 24 | ✓ | — |
| P015 | F | 28 | ✓ | ✓ |
| P016 | F | 19 | ✓ | — |
| P017 | M | 40 | ✓ | — |
| P018 | F | 21 | ✓ | — |
| P019 | F | 27 | ✓ | — |
| P020 | M | 41 | ✓ | — |
| P021 | F | 23 | — | ✓ |
| P022 | F | 23 | — | ✓ |
| P023 | F | 23 | — | ✓ |
| P024 | F | 18 | — | ✓ |
| P025 | F | 23 | — | ✓ |
| P026 | M | 22 | — | ✓ |
| P027 | M | 25 | — | ✓ |


### 2.1 Model Development Dataset
#### 2.1.1 General Description
The Model Development Dataset (MD) consists of 20 healthy subjects, comprising 14 women and six men, aged between 19 and 
54 years (27 $\pm$ 3.7 years). The dataset contains nine sub-activities that can be associated to three main activities,
namely sitting, standing, and walking. The acquired sub-activities and their correspondence to the respective main 
activity are shown below.

<p align="center">
<img src="./figures/HTA.svg" alt="test" width="600">
</p>

#### 2.1.2 Dataset Structure
The MD is organized hierarchically by participant and activity-related recording sessions.
```text
model_development/
├── P001/
│   ├── cabinets/
│   │   ├── opensignals_ANDROID_ACCELEROMETER_2024-04-08_17-37-29.txt
│   │   ├── opensignals_ANDROID_GYROSCOPE_2024-04-08_17-37-29.txt
│   │   ├── opensignals_ANDROID_MAGNETIC_FIELD_2024-04-08_17-37-29.txt
│   │   └── opensignals_ANDROID_ROTATION_VECTOR_2024-04-08_17-37-29.txt
│   ├── sitting/
│   │   ├── ...
│   ├── stairs/
│   │   ├── ...
│   ├── standing/
│   │   ├── ...
│   └── walking/
│       ├── ...
├── P002/
│   ├── .../
│   ├── .../
│   ├── .../
│   ├── .../
│   └── .../
├── .../
```
__Activity-Related Recording Folders:__  
Within each participant folder:
* Separate folders are provided for recordings that contain activities that thematically have been recorded together:  `cabinets/`, `sitting/`, `stairs/`, `standing/`, and `walking/`
* Each folder contains the raw smartphone sensor signals recorded during that recording, stored as `.txt` files.

__Activity-Related Recordings:__  
Similar sub-activities were grouped together to form five separate sessions. Each session included clearly defined 
sections in which the subject was instructed to perform a specific sub-activity. 
1. At the beginning of each recording session, each subject performed roughly ten jumps for synchronization purposes. 

To facilitate the segmentation of the sub-activities performed in the same recording, short pauses were introduced in 
between. These pauses were
2. ten-seconds stops (no movement) for activities associated with walking.
3. ten-seconds stops with a jump in the middle for activities performed while standing. 

The described segmentation patterns are shown in the figure below.

<p align="center">
<img src="./figures/acquisition_protocol.svg " alt="Segmentation patterns within a walking (top) and a standing (bottom) 
recording: (1) synchornisation jumps, (2) ten-second stop, (3) ten-second stop with jump in the middle." width="800">
</p>

__Sensor Data Files:__  
Each session sub-folder contains the raw smartphone sensor signals stores as `.txt` files. The naming convention for 
sensor files is `opensignals_ANDROID_{SENSOR_NAME}_YYYY-MM-DD_HH-MM-SS.txt`, where `{SENSOR_NAME}` corresponds to one of the 
following smartphone sensors:
* `ACCELEROMETER`
* `GYROSCOPE`
* `MAGNETIC_FIELD`
* `ANDROID_ROTATION_VECTOR`

### 2.2 Model Evaluation Dataset
#### 2.2.1 General Description
The model Evaluation Dataset (ME) contains data from 13 healthy subjects, eight women and five men, aged between 18 and 
34 years (24.2 $\pm$ 3.7 years). For this dataset, subjects carried out the daily work activities for roughly six hours.
No specific instructions were given to the subjects during the data acquisition. Instead, they were encouraged to carry 
out their daily work tasks while labeling the three main activities: sitting, standing, and walking, using an Android 
application developed for this study. The application generates a .txt file that contains the time, in the format 
"HH:MM:SS.ms", and the corresponding activity that is logged via a button press. Participants were indicated to press 
the corresponding activity button once an activity was initiated. A screenshot of the application is provided below.

<p align="center">
<img src="./figures/logger_app.svg" alt="App for logging performed activities during acquisition of ME dataset." 
width="200">
</p>

Given the delay that results from reaction times of pressing the button, and the potential of forgetting to label or 
mislabeling, the generated files were inspected and corrected by two separate experts by visualizing the labels together
with the corresponding signals, namely the y-axis of the ACC. The correction consisted of aligning the labels with the 
onset of each activity.

#### 2.2.2 Dataset Structure
The ME dataset is organized hierarchically by participant and recording session.

__Directory Layout:__
```text
model_evaluation/
├── P001/
│   ├── 2025-05-26_09-50-02/
│   │   ├── opensignals_ANDROID_ACCELEROMETER_2025-05-26_09-50-02.txt
│   │   ├── opensignals_ANDROID_GYROSCOPE_2025-05-26_09-50-02.txt
│   │   ├── opensignals_ANDROID_MAGNETIC_FIELD_2025-05-26_09-50-02.txt
│   │   └── opensignals_ANDROID_ROTATION_VECTOR_2025-05-26_09-50-02.txt
│   └── 20250526_activity_log.txt
├── P002/
│   ├── .../
│   └── ...
├── .../
```
__Recording Sessions:__  
Within each participant folder:
* Each recording session is stored in a sub-folder named using the format `YYYY-MM-DD_HH-MM-SS`, corresponding to the date and start time of the recording (e.g., `2025-05-26_09-50-02`).
* A session-level label file named `YYYYMMDD_activity_log.txt` (e.g., `20250526_activity_log.txt`) contains the activity annotations for that participant.

__Sensor Data Files:__  
Each session sub-folder contains the raw smartphone sensor signals stores as `.txt` files. The naming convention for 
sensor files is `opensignals_ANDROID_{SENSOR_NAME}_YYYY-MM-DD_HH-MM-SS.txt`, where `{SENSOR_NAME}` corresponds to one of the 
following smartphone sensors:
* `ACCELEROMETER`
* `GYROSCOPE`
* `MAGNETIC_FIELD`
* `ANDROID_ROTATION_VECTOR`




