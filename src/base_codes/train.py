import os, pickle
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

import config
from utils import setup_logger, ModelFactory

def train(X_train, y_train, model_config, logger):
    model = ModelFactory(config.MODEL_NAME, model_config, logger)
    model.fit(X_train, y_train)

    return model

def valid(model, X_test, y_test):
    pred = model.predict(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, pred)
    return auc_score

if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logger = setup_logger('./logs/train_{0}.log'.format(NOW))
    df = pd.read_pickle(os.path.join(config.SAVE_PATH, 'application_train.pickle'))
    logger.info('train_df shape: {0}'.format(df.shape))
    X = df[[col for col in df.columns if col not in config.unused]]
    y = df.TARGET

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)

    model = train(X_train, y_train, config.MODEL_CONFIG, logger)
    auc_score = valid(model, X_test, y_test)
    logger.info('AUC Score: {0}'.format(auc_score))
    logger.info('Save Model to directory {0}'.format(config.SAVE_PATH))

    pickle.dump(model, open(os.path.join(config.SAVE_PATH, config.MODEL_FILE), 'wb'))