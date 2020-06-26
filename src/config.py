import psutil
import pandas as pd

SEED = 42
DATA_PATH = '../data'#./inputに変更する必要あり。
SAVE_PATH = './output'
MODEL_NAME = 'LogisticRegression'
MODEL_FILE = '{0}_model.pickle'.format(MODEL_NAME)
TEST_SIZE = 0.2
PERCENT = 0.5
SCALING = True

########## M5 new constants ##########
VER = 1                          # Our model version
SEED = 42                        # We want all things
N_CORES = psutil.cpu_count()     # Available CPU cores

#PATHS for Features
ORIGINAL = '../data/input/m5-forecasting-accuracy/'
BASE     = '../data/input/m5-simple-fe/grid_part_1.pkl'
PRICE    = '../data/input/m5-simple-fe/grid_part_2.pkl'
CALENDAR = '../data/input/m5-simple-fe/grid_part_3.pkl'
LAGS     = '../data/input/m5-lags-features/lags_df_28.pkl'
MEAN_ENC = '../data/input/m5-custom-features/mean_encoding_df.pkl'

#LIMITS and const
TARGET      = 'sales'            # Our target
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1913               # End day of our train set
P_HORIZON   = 28                 # Prediction horizon

#STORES ids
STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())

unused = ['sales', 'TARGET']

MODEL_CONFIG = {
    'LogisticRegression': {
        'C': 1.0,
        'random_state': SEED,
        'max_iter': 100,
        'penalty': 'l2',
        'n_jobs': -1,
        'solver': 'lbfgs',
        #'class_weight': {0:1, 1:2},
    },
    'RandomForest': {
        'max_depth': 8,
        'min_sample_split': 2,
        'n_estimator': 200,
        'random_state': SEED,
        'class_weight': {0:1, 1:2},
    },
    'LightGBM': {
        'boosting_type': 'gbdt',
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'subsample': 0.5,
        'subsample_freq': 1,
        'learning_rate': 0.03,
        'num_leaves': 2**11-1,
        'min_data_in_leaf': 2**12-1,
        'feature_fraction': 0.5,
        'max_bin': 100,
        'n_estimators': 1400,
        'boost_from_average': False,
        'verbose': -1,
    }
}