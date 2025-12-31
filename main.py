# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import argparse
from pathlib import Path
import numpy as np

# internal imports
from constants import (VALID_SENSORS, SEGMENTED_DATA_FOLDER, EXTRACTED_FEATURES_FOLDER,
                       RESULTS_FOLDER, MODEL_DEVELOPMENT_FOLDER , MODEL_EVALUATION_FOLDER,
                       RANDOM_SEED)
from raw_data_processor import generate_segmented_dataset
from feature_extractor import extract_features
from HAR import perform_model_configuration, test_production_models
from file_utils import create_dir




# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
# definition of folder_path
RAW_DATA_FOLDER_PATH_MD = r'E:\Prevoccupai_HAR\model_development' #   E:\Prevoccupai_HAR\subject_data\raw_signals_backups\acquisitions
RAW_DATA_FOLDER_PATH_ME = r'E:\Prevoccupai_HAR\work_simulation\raw_data' #    E:\Prevoccupai_HAR\model_evaluation_corrected
OUTPUT_FOLDER_PATH = r'E:\Prevoccupai_HAR\HAR_JA_article' # E:\Prevoccupai_HAR\test

# definition of window size and sampling rate
W_SIZE_S = 5.0
FS = 100
LABEL_MAP = {'sitting': 0, 'standing': 1, 'walking': 2}

# ------------------------------------------------------------------------------------------------------------------- #
# argument parsing
# ------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()

# (1) paths
parser.add_argument('--MD_dataset_path', default=RAW_DATA_FOLDER_PATH_MD, help='Path to model development dataset.')
parser.add_argument('--ME_dataset_path', default=RAW_DATA_FOLDER_PATH_ME, help='Path to model evaluation dataset.')
parser.add_argument('--output_path', default=OUTPUT_FOLDER_PATH, help='Path to output folder.')

# (2) dataset parameters
parser.add_argument('--fs', default=100, type=int, help="The sampling frequency used during data acquisition.")
parser.add_argument('--window_size_s', default=5.0, type=float, help='The window size (in seconds) for signal windowing.')
parser.add_argument('--load_sensors', nargs="+", default=VALID_SENSORS, help="The sensor to be loaded (as List[str]), e.g., [\"ACC\", \"GYR\"].")
parser.add_argument('--window_scaler', default=None, choices=['minmax', 'standard', None], help="The scaling that should be applied to each data window.")
parser.add_argument('--balancing_type', default='main_classes', choices=['main_classes', 'sub_classes'], help="The balancing type to use for data balancing.")
parser.add_argument('--output_file_type', default='.npy', choices=['.npy', '.csv'], help="The output file type for data segmentation and feature extraction.")
parser.add_argument('--input_file_type', default='.npy', choices=['.npy', '.csv'], help="The input file type for feature extraction. Should match the file type that was chosen for data segmentation.")

# (3) plotting for verifying proper segmentation of dataset
parser.add_argument('--plot_cropped_tasks', default=False, type=bool, help="Whether or not to plot the cropped tasks after segmentation.")
parser.add_argument('--plot_segment_lines', default=False, type=bool, help="Whether or not to plot where the data will be segmented by indicating segmentation lines on top of the y-acc signal.")

# parse the provided arguments
parsed_args = parser.parse_args()


# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    # obtain parameters that are needed for all steps
    w_size = parsed_args.window_size_s
    fs = parsed_args.fs
    output_path = Path(parsed_args.output_path)
    output_file_type = parsed_args.output_file_type

    # check whether the segmented dataset has been generated
    if not os.path.exists(os.path.join(output_path, SEGMENTED_DATA_FOLDER)):
        print("\n# ----------------------------------------------- #")
        print("# ------- 1. generating segmented dataset ------- #")
        print("# ----------------------------------------------- #")

        # parse the necessary inputs
        md_dataset_path = Path(parsed_args.MD_dataset_path)
        load_sensors = parsed_args.load_sensors
        plot_segment_lines = parsed_args.plot_segment_lines
        plot_cropped_tasks = parsed_args.plot_cropped_tasks

        # generate the segmented data set
        generate_segmented_dataset(md_dataset_path, segmented_data_path=output_path, load_sensors=load_sensors,
                                   fs=fs, output_file_type=output_file_type,
                                   plot_cropped_tasks=plot_cropped_tasks,
                                   plot_segment_lines=plot_segment_lines)

    # check whether the feature dataset has been generated
    if not os.path.exists(os.path.join(output_path, EXTRACTED_FEATURES_FOLDER)):
        print("\n# -------------------------------------- #")
        print("# ------- 2. extracting features ------- #")
        print("# -------------------------------------- #")

        # parse necessary inputs
        window_scaler = parsed_args.window_scaler
        input_file_type = parsed_args.input_file_type

        # path to segmented data folder
        segmented_data_path = os.path.join(output_path, SEGMENTED_DATA_FOLDER)

        # extract features and save them to individual subject files
        extract_features(segmented_data_path, features_data_path=output_path,
                         window_size=w_size, window_scaler=window_scaler,
                         default_input_file_type=input_file_type, output_file_type=output_file_type)

    # set random seed
    np.random.seed(RANDOM_SEED)

    # path to folder containing the feature files for the different normalization types
    feature_data_path = os.path.join(output_path, EXTRACTED_FEATURES_FOLDER)

    # parse necessary inputs
    me_dataset_path = Path(parsed_args.ME_dataset_path)
    load_sensors = parsed_args.load_sensors
    balancing_type = parsed_args.balancing_type

    # generate results folder
    results_folder = create_dir(output_path, RESULTS_FOLDER)

    # Obtain th best KNN, SVN, and RF models and train for production
    if not os.path.exists(os.path.join(output_path, RESULTS_FOLDER, MODEL_DEVELOPMENT_FOLDER)):
        print("\n# -------------------------------------------------------------------------------------------------------------------------- #")
        print("# ------- 3. Training and Evaluating different models (Random Forest vs. KNN vs. SVM) on model development dataset ------- #")
        print("# -------------------------------------------------------------------------------------------------------------------------- #")
        perform_model_configuration(feature_data_path, results_folder,  balancing_type=balancing_type,
                                    window_size_samples=int(W_SIZE_S * FS))

    # Test the production models KNN, SVM, RF on the real world dataset
    if not os.path.exists(os.path.join(output_path, RESULTS_FOLDER, MODEL_EVALUATION_FOLDER)):
        print("\n# -------------------------------------------------------------------------- #")
        print("# ------- 4. testing the trained models on model development dataset ------- #")
        print("# -------------------------------------------------------------------------- #")
        test_production_models(me_dataset_path, results_folder, LABEL_MAP, fs=fs, w_size_sec=w_size,
                               load_sensors=load_sensors)

