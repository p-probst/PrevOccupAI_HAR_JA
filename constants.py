"""
file for defining constants within the entire project
"""

# ------------------------------------------------------------------------------------------------------------------- #
# sensor constants
# ------------------------------------------------------------------------------------------------------------------- #
# definition of valid sensors (for now only phone sensors)
ACC = 'ACC'
GYR = 'GYR'
MAG = 'MAG'
ROT = 'ROT'

# definition of valid activities sensors (to ensure loading of proper files)
VALID_SENSORS = [ACC, GYR, MAG, ROT]
IMU_SENSORS = [ACC, GYR, MAG]

# mapping of valid sensors to sensor filename
SENSOR_MAP = {ACC: 'ANDROID_ACCELEROMETER',
              GYR: 'ANDROID_GYROSCOPE',
              MAG: 'ANDROID_MAGNETIC_FIELD',
              ROT: 'ANDROID_ROTATION_VECTOR'}

# ------------------------------------------------------------------------------------------------------------------- #
# activity constants
# ------------------------------------------------------------------------------------------------------------------- #
# folder names containing the acquisitions
SIT = 'sitting'
STAND = 'standing'
CABINETS = 'cabinets'  # belongs to stand activity/class
STAIRS = 'stairs'  # belongs to walk activity/class
WALK = 'walking'

# sub-activities
SIT_SUB = 'sit'
STILL = 'still'
TALK = 'talk'
COFFEE = 'coffee'
FOLDERS = 'folders'
SLOW = 'slow'
MEDIUM = 'medium'
FAST = 'fast'
UP = 'up'
DOWN = 'down'

# definition of valid activities (to ensure loading of proper folders)
VALID_ACTIVITIES = [SIT, STAND, CABINETS, WALK, STAIRS]

# mapping of recordings contained in each activity folder and the activities performed within each recording
ACTIVITY_MAP = {SIT: [f'_{SIT_SUB}'],
                STAND: [f'_{STILL}_1', f'_{TALK}', f'_{STILL}_2'],
                CABINETS: [f'_{COFFEE}', f'_{FOLDERS}'],
                WALK: [f'_{SLOW}', f'_{MEDIUM}', f'_{FAST}'],
                STAIRS: [f'_{UP}_1', f'_{DOWN}_1', f'_{UP}_2', f'_{DOWN}_2',
                         f'_{UP}_3', f'_{DOWN}_3', f'_{UP}_4', f'_{DOWN}_4']
                }


# ------------------------------------------------------------------------------------------------------------------- #
# class constants
# ------------------------------------------------------------------------------------------------------------------- #
# activity main classes
CLASS_SIT = 0
CLASS_STAND = 1
CLASS_WALK = 2

MAIN_ACTIVITY_LABELS = [CLASS_SIT, CLASS_STAND, CLASS_WALK]

# definition of label map for loading labels from activity logs of model evaluation dataset
MAIN_LABEL_MAP = {SIT: CLASS_SIT, STAND: CLASS_STAND, WALK: CLASS_WALK}


# activity sub-classes
CLASS_STAND_STILL = 3
CLASS_STAND_TALK = 4
CLASS_STAND_COFFEE = 5
CLASS_STAND_FOLDERS = 6
CLASS_WALK_SLOW = 7
CLASS_WALK_MEDIUM = 8
CLASS_WALK_FAST = 9
CLASS_WALK_STAIRS_UP = 10
CLASS_WALK_STAIRS_DOWN = 11

SUB_ACTIVITIES_STAND_LABELS = [CLASS_STAND_STILL, CLASS_STAND_TALK, CLASS_STAND_COFFEE, CLASS_STAND_FOLDERS]
SUB_ACTIVITIES_WALK_LABELS = [CLASS_WALK_SLOW, CLASS_WALK_MEDIUM, CLASS_WALK_FAST, CLASS_WALK_STAIRS_UP, CLASS_WALK_STAIRS_DOWN]

# mapping of activity folders to the main class and their respective sub-classes
MAIN_CLASS_KEY = 'main_class'
ACTIVITY_MAIN_SUB_CLASS = \
    {SIT: {MAIN_CLASS_KEY: CLASS_SIT, SIT_SUB: CLASS_SIT},
     STAND: {MAIN_CLASS_KEY: CLASS_STAND, STILL: CLASS_STAND_STILL, TALK: CLASS_STAND_TALK},
     CABINETS: {MAIN_CLASS_KEY: CLASS_STAND, COFFEE: CLASS_STAND_COFFEE, FOLDERS: CLASS_STAND_FOLDERS},
     WALK: {MAIN_CLASS_KEY: CLASS_WALK, SLOW: CLASS_WALK_SLOW, MEDIUM: CLASS_WALK_MEDIUM, FAST: CLASS_WALK_FAST},
     STAIRS: {MAIN_CLASS_KEY: CLASS_WALK, UP: CLASS_WALK_STAIRS_UP, DOWN: CLASS_WALK_STAIRS_DOWN}
     }
# ------------------------------------------------------------------------------------------------------------------- #
# supported file types
# ------------------------------------------------------------------------------------------------------------------- #
CSV = '.csv'
NPY = '.npy'
TXT = '.txt'

# definition of valid file typs for storing and loading processed data. The user can decide
# whether to use .csv or .npy files based on their preferences
VALID_FILE_TYPES = [CSV, NPY]

# ------------------------------------------------------------------------------------------------------------------- #
# output folder names
# ------------------------------------------------------------------------------------------------------------------- #
SEGMENTED_DATA_FOLDER = 'segmented_data'
EXTRACTED_FEATURES_FOLDER = 'extracted_features'
RESULTS_FOLDER = 'results'
MODEL_DEVELOPMENT_FOLDER = 'model_development'
MODEL_EVALUATION_FOLDER = 'model_evaluation'

# ------------------------------------------------------------------------------------------------------------------- #
# json files and keys
# ------------------------------------------------------------------------------------------------------------------- #
SENSOR_COLS_JSON = 'numpy_columns.json'
LOADED_SENSORS_KEY = 'loaded_sensors'

CLASS_INSTANCES_JSON = 'class_instances.json'
FEATURE_COLS_KEY = 'feature_cols'
MAIN_LABEL_KEY = 'main_label'
SUB_LABEL_KEY = 'sub_label'

# ------------------------------------------------------------------------------------------------------------------- #
# random seed (for reproducibility)
# ------------------------------------------------------------------------------------------------------------------- #
RANDOM_SEED = 42