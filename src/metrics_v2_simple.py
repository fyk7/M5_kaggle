import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import gc

#以下のread_pickleはmain fileで読み込む必要がある?

#read_scale
sw_df = pd.read_pickle('../data/input/sw_df_1914.pkl')
S = sw_df.s.values
W = sw_df.w.values
SW = sw_df.sw.values
#del sw_df
#gc.collect()

#read_weight
roll_mat_df = pd.read_pickle('../data/input/roll_mat_df_1914.pkl')
roll_index = roll_mat_df.index
roll_mat_csr = csr_matrix(roll_mat_df.values)
#del roll_mat_df
#gc.collect()

def rollup(v):
    return roll_mat_csr*v

def wrmsse(preds, y_true, score_only=False, s = S, w = W, sw=SW):
    '''
    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
    sequence_length - np.array of size (42840,)
    sales_weight - sales weights based on last 28 days: np.array (42840,)
    '''
    if score_only:
        return np.sum(
                np.sqrt(
                    np.mean(
                        np.square(rollup(preds.values-y_true.values))# (30490, 28)をroll_up
                            ,axis=1)) * sw)/12 #<-used to be mistake here
    else: 
        score_matrix = (np.square(rollup(preds.values-y_true.values)) * np.square(w)[:, None])/ s[:, None]
        score = np.sum(np.sqrt(np.mean(score_matrix,axis=1)))/12 #<-used to be mistake here
        return score, score_matrix

#calcurate score from submission files
def calc_wrmsse_v2(END_TRAIN=1914, submission_path):
    sales = pd.read_csv('../data/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
    sub = pd.read_csv(submission_path)
    sub = sub[sub.id.str.endswith('validation')]
    sub.drop(['id'], axis=1, inplace=True)

    DAYS_PRED = sub.shape[1]    # 28
    # dayCols = ["d_{}".format(i) for i in range(1914-DAYS_PRED, 1914)]
    dayCols = ["d_{}".format(i) for i in range(END_TRAIN - DAYS_PRED, END_TRAIN)]
    y_true = sales[dayCols]


    score = wrmsse(sub, y_true, score_only=True)
    print(score)