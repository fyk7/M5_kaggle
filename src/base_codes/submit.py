import os, pickle
import datetime
import pandas as pd

import config
from utils import setup_logger, ModelFactory

def test(model, X_test, submission):
    pred = model.predict(X_test)[:, 1]
    sub = pd.DataFrame({'SK_ID_CURR': submission.SK_ID_CURR , 'TARGET': pred})
    sub.to_csv('submit.csv', index = False)

if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logger = setup_logger('./logs/submit_{0}.log'.format(NOW))

    logger.info('Loading Test Data')
    df = pd.read_pickle(os.path.join(config.SAVE_PATH, 'application_test.pickle'))
    X = df[[col for col in df.columns if col not in config.unused]]
    submission = pd.read_csv('../sample_submission.csv.zip')

    logger.info('Loading Pre-Training Model')
    model = pickle.load(open(os.path.join(config.SAVE_PATH, config.MODEL_FILE), 'rb'))

    test(model, X, submission)