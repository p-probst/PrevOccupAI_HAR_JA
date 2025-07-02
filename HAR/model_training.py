# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.base import ClassifierMixin, BaseEstimator
from typing import Dict, Any, List, Union

# internal imports
from .load import load_features
from .feature_selection import remove_low_variance, remove_highly_correlated_features, select_k_best_features
from constants import RANDOM_SEED
from file_utils import create_dir

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
RF = "Random Forest"
SVM = "SVM"
KNN = "KNN"

STD_STEP = 'std'
PARAM_GRID = 'param_grid'
ESTIMATOR = 'estimator'


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def perform_model_configuration(data_path: str, balancing_type: str, window_size_samples: int) -> None:
    """
    Evaluates 3 different models (Random Forest, KNN, and SVM)  using a nested cross-validation to select which of
    these models is used for production. The model selection is only performed on the training data.
    :param data_path: the path to the data. This should point to the folder containing the extracted features.
    :param balancing_type: the data balancing type. Can be either:
                         'main_classes': for balancing the data in such a way that each main class has the (almost) the
                                       same amount of data. This ensures that each sub-class within the main class has
                                       the same amount of instances.
                         'sub_classes': for balancing that all sub-classes have the same amount of instances
                         None: no balancing applied. Default: None
    :param window_size_samples: the number of samples per window. Used for creating folder and file names.
    :return: None
    """
    # define model dictionary
    model_dict = {

        KNN: {ESTIMATOR: Pipeline([(STD_STEP, StandardScaler()), (KNN, KNeighborsClassifier(algorithm='ball_tree'))]),
              PARAM_GRID: [{f'{KNN}__n_neighbors': list(range(1, 15)), f'{KNN}__p': [1, 2]}]},

        SVM: {ESTIMATOR: Pipeline([(STD_STEP, StandardScaler()), (SVM, SVC(random_state=RANDOM_SEED))]), PARAM_GRID: [
            {f'{SVM}__kernel': ['rbf'], f'{SVM}__C': np.power(10., np.arange(-4, 4)),
             f'{SVM}__gamma': np.power(10., np.arange(-5, 0))},
            {f'{SVM}__kernel': ['linear'], f'{SVM}__C': np.power(10., np.arange(-4, 4))}]},

        RF: {ESTIMATOR: RandomForestClassifier(random_state=RANDOM_SEED), PARAM_GRID: [
            {"criterion": ['gini', 'entropy'], "n_estimators": [50, 100, 500, 1000], "max_depth": [2, 5, 10, 20, 30]}]}
    }

    # path to feature folder (change the folder name to run the different normalization schemes)
    feature_data_folder = os.path.join(data_path, f"w_{window_size_samples}_sc_none")

    # load feature, labels, and subject IDs
    X, y_main, y_sub, subject_ids = load_features(feature_data_folder, balance_data=balancing_type)

    # split of train and test set
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y_main, groups=subject_ids))

    # get train and test sets
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    print(f"Total number of instances for training: {X_train.shape[0]}")

    # TODO: we only consider main_classes balancing, no need to handle it here.
    # get y depending on the balancing type
    if balancing_type == 'main_classes':
        y_train = y_main.iloc[train_idx]
        y_test = y_main.iloc[test_idx]

    else:  # sub-class balancing
        y_train = y_sub[train_idx]
        y_test = y_sub[test_idx]

        # add label encoding, as in this case the labels are non-consecutive
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train))
        y_test = pd.Series(le.fit_transform(y_test))

    print(f"subjects train: {subject_ids[train_idx].unique()}")
    print(f"subjects test: {subject_ids[test_idx].unique()}")

    # get the subjects for training (and testing)
    subject_ids_train = subject_ids.iloc[train_idx]

    # cycle over the models
    for model_name, param_dict in model_dict.items():
        print('### ----------------------------------------- ###')
        print(f'Algorithm: {model_name}')

        # hyperparameter tuning per model
        _hyperparameter_tuning(model_name, param_dict, subject_ids_train, X_train, y_train, X_test, y_test,
                               window_size_samples=window_size_samples)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _hyperparameter_tuning(model_name: str, param_dict:  Union[List[Dict[str, Any]], Dict[str, Any]],
                           subject_ids_train: pd.Series, X_train_all: pd.DataFrame, y_train: pd.Series,
                           X_test_all: pd.DataFrame, y_test: pd.Series,
                           window_size_samples: int, cv_splits: int = 5) -> None:

    # TODO: update
    """
    Evaluates multiple machine learning models using nested cross-validation.

    This function initializes and evaluates different models (SVM, KNN, and Random Forest)
    based on the specified normalization type. It applies nested cross-validation to assess
    model performance and saves the results.

    :param estimator:
    :param param_grid:
    :param X_train: pandas.DataFrame containing the training data
    :param y_train: pandas.Series containing the labels
    :param window_size_samples: the number of samples per window. Used for creating folder and file names.
    :return: None
    """
    # get the path to the current project
    project_path = os.getcwd()

    # create results directory (if it doesn't exist)
    folder_path = create_dir(project_path, "Results")

    # init best accuracy
    best_acc = 0

    # dict for holding results
    results_list = []

    # get the estimator and the param grid
    est = param_dict[ESTIMATOR]
    param_grid_est = param_dict[PARAM_GRID]

    # test different number of features
    for num_features_retain in [5, 10, 15, 20, 25, 30, 35]:
        print("\n.................................................................")
        print(f"Classes used: {np.unique(y_train)}")
        print(f"Testing {num_features_retain} features...\n")

        # perform model agnostic feature selection
        X_train, X_test = remove_low_variance(X_train_all, X_test_all, threshold=0.1)
        X_train, X_test = remove_highly_correlated_features(X_train, X_test, threshold=0.9)
        X_train, X_test = select_k_best_features(X_train, y_train, X_test, k=num_features_retain)

        print(f"Used features: {X_train.columns.values}")

        # Perform Grid Search
        cv = GroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
        grid_search = GridSearchCV(estimator=est, param_grid=param_grid_est, scoring='accuracy', cv=cv, n_jobs=-1)
        grid_search.fit(X_train, y_train, groups=subject_ids_train)

        # get best estimator of CV
        best_model = grid_search.best_estimator_

        # get the params of best estimator
        best_parameters = grid_search.best_params_

        # evaluate estimator
        # (1) train accuracy
        train_acc = accuracy_score(y_true=y_train, y_pred=best_model.predict(X_train))

        # (2) test accuracy
        test_acc = accuracy_score(y_true=y_test, y_pred=best_model.predict(X_test))

        # print the results
        print("\nResults of hyperparameter tuning")
        print(f"best score (CV avg.): {grid_search.best_score_ * 100: .2f}")
        print(f"best parameters: {best_parameters}")

        print("\nResults of best model.")
        print(f"train accuracy: {train_acc * 100: .2f}")
        print(f"test accuracy: {test_acc * 100: .2f}")

        # store the results
        results_list.append({"num_features": num_features_retain, "train_acc": train_acc, "test_acc": test_acc,
                             "best_param": str(best_parameters)})

        # store the best model
        if test_acc > best_acc:

            # save the best model over all feature sets
            joblib.dump(best_model, os.path.join(folder_path, f"{model_name}.joblib"))

            # update the accuracy
            best_acc = test_acc

    # create DataFrame with results
    results_df = pd.DataFrame(results_list)

    # store the DataFrame
    results_df.to_csv(os.path.join(folder_path, f"{model_name}_results.csv"))


