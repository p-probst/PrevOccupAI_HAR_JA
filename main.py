# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os

# internal imports
from constants import VALID_SENSORS, SEGMENTED_DATA_FOLDER, EXTRACTED_FEATURES_FOLDER, RANDOM_SEED
from raw_data_processor import generate_segmented_dataset
from feature_extractor import extract_features
from HAR import perform_model_configuration, test_production_models
import numpy as np


# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GENERATE_SEGMENTED_DATASET = False
EXTRACT_FEATURES = False
ML_HAR = True
ML_MODEL_TUNING = False
ML_TEST_PRODUCTION_MODELS = True

# definition of folder_path
RAW_DATA_FOLDER_PATH = 'G:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data\\raw_signals_backups\\acquisitions'
OUTPUT_FOLDER_PATH = 'G:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data'


# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    if GENERATE_SEGMENTED_DATASET:

        # generate the segmented data set
        generate_segmented_dataset(RAW_DATA_FOLDER_PATH, OUTPUT_FOLDER_PATH, load_sensors=VALID_SENSORS,
                                   fs=100, output_file_type='.npy', plot_cropped_tasks=False, plot_segment_lines=False)

    if EXTRACT_FEATURES:

        print("extracting features")

        # path to segmented data folder
        segmented_data_path = os.path.join(OUTPUT_FOLDER_PATH, SEGMENTED_DATA_FOLDER)

        # extract features and save them to individual subject files
        extract_features(segmented_data_path, OUTPUT_FOLDER_PATH, window_scaler='standard',
                         default_input_file_type='.npy',
                         output_file_type='.npy')

    if ML_HAR:

        # setting variables for run
        balancing_type = 'main_classes'

        print("HAR model training/test")

        # set random seed
        np.random.seed(RANDOM_SEED)

        # path to folder containing the feature files for the different normalization types
        feature_data_path = os.path.join(OUTPUT_FOLDER_PATH, EXTRACTED_FEATURES_FOLDER)

        # perform model selection
        if ML_MODEL_TUNING:

            print("Evaluating different models (Random Forest vs. KNN vs. SVM)")
            perform_model_configuration(feature_data_path, balancing_type=balancing_type, window_size_samples=500)

        #train production model using the number of features and the normalization type found through model selection
        if ML_TEST_PRODUCTION_MODELS:
            raw_data_path = "G:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\work_simulation\\raw_data"
            label_map = {'sitting': 0, 'standing': 1, 'walking': 2}

            print("\ntesting the production models")
            test_production_models(raw_data_path, label_map, fs=100, w_size_sec=5.0, load_sensors=VALID_SENSORS)