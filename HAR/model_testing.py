"""
Functions for testing the production models on the real-world dataset.

Available Functions
-------------------
[Public]
test_production_models(...): Tests the production models (KNN, SVM, RF) on the real-world data
-------------------
[Private]
_test_production_model(...): Loads the production model obtains the predictions.
_expand_classification(...): Expands the predictions to the size of the original signal.
_plot_all_predictions(...): Generates and saves a figure with the true labels and predictions over time.
-------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd
import numpy as np
import tsfel
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# internal imports
from constants import TXT
from raw_data_processor import load_data_from_same_recording
from .load import load_labels_from_log, load_production_model
from feature_extractor import pre_process_signals, window_and_extract_features
from file_utils import create_dir

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
RF = "Random Forest"
SVM = "SVM"
KNN = "KNN"

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def test_production_models(raw_data_path: str, label_map: Dict[str, int], fs: int, w_size_sec: float,
                           load_sensors: Optional[List[str]] = None) -> None:
    """
    Tests the production models ("KNN", "SVM", "RF") on real-world data.

    This function loads the real-world dataset and corresponding labels, pre-processes the data, and extracts
    the features used during model training. It then uses the production models to predict the activities,
    saves the evaluation metrics to a CSV file, and generates a plot showing the predictions over time.

    :param raw_data_path: path to the folder containing the subject folders (i.e. "...\raw_data_path\P001\signals.txt")
    :param label_map: A dictionary mapping activity strings to numeric labels.
                        e.g., {"sitting": 1, "standing": 2, "walking": 3}.
    :param fs: the sampling frequency
    :param w_size_sec: the window size in seconds
    :param load_sensors: list of sensors (as strings) indicating which sensors should be loaded.
                        Default: None (all sensors are loaded)
    :return: None
    """
    # list for holding the metrics for all subjects
    results_list = []

    # list all folders within the raw_data_path
    subject_folders = os.listdir(raw_data_path)

    # get the folders that contain the subject data. Subject data folders start with 'S' (e.g., S001)
    subject_folders = [folder for folder in subject_folders if folder.startswith('P')]

    for subject in subject_folders:
        print("\n#----------------------------------------------------------------------#")
        print(f"# Processing data for Subject {subject}")

        # innit dictionary to store the results of each subject
        results_dict = {'Subject': subject}

        # get the path to the subject folder
        subject_folder_path = os.path.join(raw_data_path, subject)

        # list all files/folders inside the subject_folder
        folder_items = os.listdir(subject_folder_path)

        # get the folder containing the signals
        signals_path = [item for item in folder_items if os.path.isdir(os.path.join(subject_folder_path, item))]

        # path to the folder containing the data
        data_folder_path = os.path.join(subject_folder_path, signals_path[0])

        # get the txt file containing the labels
        txt_files = [item for item in folder_items if item.endswith(TXT)]

        # get the labels file path
        txt_file_path = os.path.join(subject_folder_path, txt_files[0])

        # load sensor data into a pandas dataframe
        subject_data = load_data_from_same_recording(data_folder_path, load_sensors, fs=fs)

        # generate label vector
        labels = load_labels_from_log(txt_file_path, label_map, subject_data.shape[0])

        # add the labels to the dataframe
        subject_data['labels'] = labels

        # get the sensor names
        sensor_names = subject_data.columns.values[1:-1]

        # pre process signals
        sensor_data, true_labels = pre_process_signals(subject_data, sensor_names, w_size=w_size_sec, fs=fs)

        # extract features
        cfg = tsfel.load_json(f".\\HAR\\cfg_file_production_models.json")
        features_df = window_and_extract_features(sensor_data, sensor_names, cfg, w_size_sec=w_size_sec, fs=fs)

        # get a list with the models to be tested
        models_list = [KNN, SVM, RF]

        # list for holding the accuracies
        acc_list = []

        # list for holding the precisions
        precisions_list = []

        # list for holding the recalls of the models
        recalls_list = []

        # list for holding the f1 scores of the models
        f1_list = []

        # list for holding the predictions
        predictions_list = []

        for model in models_list:

            # generate the model paths
            model_path = os.path.join(os.getcwd(),"Results", f"{model}.joblib")

            # get predictions
            predictions = _test_production_model(model_path, features_df, w_size_sec, fs)

            # calculate metrics
            acc = round(accuracy_score(true_labels, predictions) * 100, 2)
            precision = round(precision_score(true_labels, predictions, average='weighted')*100, 2)
            recall = round(recall_score(true_labels, predictions, average='weighted')*100, 2)
            f1 = round(f1_score(true_labels, predictions, average='weighted')*100, 2)

            # append results to the lists
            acc_list.append(acc)
            precisions_list.append(precision)
            recalls_list.append(recall)
            f1_list.append(f1)
            predictions_list.append(predictions)

        for n in range(len(models_list)):

            # add model and metrics to the results dict
            results_dict.update({f"acc_{models_list[n]}": acc_list[n], f"precision_{models_list[n]}": precisions_list[n],
                                 f"recall_{models_list[n]}": recalls_list[n], f"f1_score_{models_list[n]}": f1_list[n]})

        # append the results dictionary to the list with the results of all subjects
        results_list.append(results_dict)

        # generate the plots with the models predictions for each subject
        _plot_all_predictions(true_labels, predictions_list, acc_list, models_list, subject)

    # create DataFrame with results
    results_df = pd.DataFrame(results_list)

    # store the DataFrame
    results_df.to_csv(os.path.join(os.getcwd(),"Results", "metrics_results.csv"))


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _test_production_model(model_path: str, features: pd.DataFrame, w_size_sec: float, fs: int) -> List[int]:
    """
    Loads the production model obtains the predictions.
    :param model_path: path to the model
    :param features: pandas DataFrame containing the features
    :param w_size_sec: the window size in seconds
    :param fs: the sampling frequency
    :return: a list containing the predictions
    """

    # load the model
    model, feature_names = load_production_model(model_path)

    # get the features that are needed for the classifier
    features = features[feature_names]

    # classify the data - vanilla model
    y_pred = model.predict(features)

    # expand the predictions to the size of the original signal
    y_pred_expanded = _expand_classification(y_pred, w_size_sec=w_size_sec, fs=fs)

    return y_pred_expanded


def _expand_classification(clf_result: np.ndarray, w_size_sec: float, fs: int) -> List[int]:
    """
    Converts the time column from the android timestamp which is in nanoseconds to seconds.
    Parameters.
    :param clf_result: list with the classifier prediction where each entry is the prediction made for a window.
    :param w_size_sec: the window size in seconds that was used to make the classification.
    :param fs: the sampling frequency of the signal that was classified.
    :return: the expanded classification results.
    """

    expanded_clf_result = []

    # cycle over the classification results list
    for i, p in enumerate(clf_result):
        expanded_clf_result += [p] * int(w_size_sec * fs)

    return expanded_clf_result


def _plot_all_predictions(labels: np.ndarray, expanded_predictions: List[List[int]], accuracies: List[float],
                          model_names: List[str], subject_id: str) -> None:
    """
    Generates and saves a figure with len(model_names) + 1 plots. The first plot corresponds to true labels over time,
    and the remaining plots correspond to the predictions of the models over time.

    :param labels: numpy.array containing the true labels
    :param expanded_predictions: List os numpy.arrays containing the predictions expanded to the size of the true label vector
    :param accuracies: List containing the accuracies of the models.
    :param model_names: list of strings pertaining to the name of the models used
    :param subject_id: str with the subject identifier
    :return: None
    """

    n_preds = len(expanded_predictions)
    fig, axes = plt.subplots(nrows=n_preds + 1, ncols=1, sharex=True, sharey=True, figsize=(30, 3 * (n_preds + 1)))
    fig.suptitle(f"True labels vs Predicted labels", fontsize=24)

    # Plot true labels
    axes[0].plot(labels, color='teal')
    axes[0].set_title("True Labels", fontsize=18)

    # Plot each prediction
    for i, (pred, acc, name) in enumerate(zip(expanded_predictions, accuracies, model_names)):
        axes[i + 1].plot(pred, color='darkorange')
        axes[i + 1].set_title(f"{name}: {acc}%", fontsize=18)

    # adjust layout
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    # get the project path
    project_path = os.getcwd()

    # generate a folder path to store the plots
    plots_output_path = create_dir(project_path,
                                   os.path.join("Results", "real_world_test", "plots"))

    # save plots
    plt.savefig(os.path.join(plots_output_path, f"results_fig_{subject_id}.png"))


