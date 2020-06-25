SEED = 42
DATA_PATH = '../data'#./inputに変更する必要あり。
SAVE_PATH = './output'
MODEL_NAME = 'LogisticRegression'
MODEL_FILE = '{0}_model.pickle'.format(MODEL_NAME)
TEST_SIZE = 0.2
PERCENT = 0.5
SCALING = True
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