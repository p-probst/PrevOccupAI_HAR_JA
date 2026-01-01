# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import argparse
from pathlib import Path
import numpy as np

# internal imports
from constants import (VALID_SENSORS, MAIN_LABEL_MAP,
                       SEGMENTED_DATA_FOLDER, EXTRACTED_FEATURES_FOLDER,
                       RESULTS_FOLDER, MODEL_DEVELOPMENT_FOLDER , MODEL_EVALUATION_FOLDER,
                       RANDOM_SEED)
from raw_data_processor import generate_segmented_dataset
from feature_extractor import extract_features
from HAR import perform_model_configuration, test_production_models
from file_utils import create_dir

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
# definition of folder_path (you can change these to the ones on your system to ensure that the path is valid)
RAW_DATA_FOLDER_PATH_MD = r'E:\Prevoccupai_HAR\model_development'
RAW_DATA_FOLDER_PATH_ME = r'E:\Prevoccupai_HAR\model_evaluation'
OUTPUT_FOLDER_PATH = r'E:\Prevoccupai_HAR\HAR_output'

NUM_SUBJECTS_MD = 20
NUM_MODELS = 3 # running KNN, SVM, and RF
NUM_FILES_ME = 14 # expecting 13 plots + the .csv file containing the results

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
parser.add_argument('--load_sensors', nargs="+", default=VALID_SENSORS, help="The sensor to be loaded (as List[str]), e.g., [\"ACC\", \"GYR\"]. The default is set to use all sensors.")
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

    # path to segmented data folder
    segmented_data_path = output_path / SEGMENTED_DATA_FOLDER

    # path to feature data folder
    feature_data_path = output_path / EXTRACTED_FEATURES_FOLDER / f'w_{int(w_size * fs)}_sc_none'

    # check whether the segmented dataset has been generated or not all subjects have been processed yet
    if not segmented_data_path.exists() or sum(1 for p in segmented_data_path.iterdir() if p.is_dir()) < NUM_SUBJECTS_MD:
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

    # check whether the feature dataset has been generated or not all subjects have been processed yet
    if not feature_data_path.exists() or sum(1 for p in feature_data_path.iterdir() if p.is_file() and p.suffix == output_file_type) < NUM_SUBJECTS_MD:
        print("\n# -------------------------------------- #")
        print("# ------- 2. extracting features ------- #")
        print("# -------------------------------------- #")

        # parse necessary inputs
        input_file_type = parsed_args.input_file_type

        # extract features and save them to individual subject files
        extract_features(segmented_data_path, features_data_path=output_path,
                         window_size=w_size,
                         default_input_file_type=input_file_type, output_file_type=output_file_type)

    # set random seed
    np.random.seed(RANDOM_SEED)

    # parse necessary inputs
    me_dataset_path = Path(parsed_args.ME_dataset_path)
    load_sensors = parsed_args.load_sensors
    balancing_type = parsed_args.balancing_type

    # generate results folder
    results_folder = create_dir(output_path, RESULTS_FOLDER)

    # path to model development results folder
    md_results_folder = output_path/ RESULTS_FOLDER /MODEL_DEVELOPMENT_FOLDER

    # path to model evaluation results folder
    me_results_folder = output_path/ RESULTS_FOLDER /MODEL_EVALUATION_FOLDER

    # train and test models (KNN, SVM, RF) on the features extracted from the model development dataset
    if not md_results_folder.exists() or sum(1 for p in md_results_folder.iterdir() if p.is_file() and p.suffix == '.joblib') < NUM_MODELS:
        print("\n# -------------------------------------------------------------------------------------------------------------------------- #")
        print("# ------- 3. Training and Evaluating different models (Random Forest vs. KNN vs. SVM) on model development dataset ------- #")
        print("# -------------------------------------------------------------------------------------------------------------------------- #")
        perform_model_configuration(feature_data_path, results_folder,  balancing_type=balancing_type,
                                    window_size_samples=int(w_size * fs))

    # test\evaluate the production models (KNN, SVM, RF) on the model evaluation (real world office work) dataset
    if not me_results_folder.exists() or sum(1 for p in me_results_folder.iterdir() if p.is_file()) < NUM_FILES_ME:
        print("\n# -------------------------------------------------------------------------- #")
        print("# ------- 4. testing the trained models on model development dataset ------- #")
        print("# -------------------------------------------------------------------------- #")
        test_production_models(me_dataset_path, results_folder, MAIN_LABEL_MAP, fs=fs, w_size_sec=w_size,
                               load_sensors=load_sensors)


    print(f"All done! The results are available at {results_folder}.")
    # ---- program end ---- #