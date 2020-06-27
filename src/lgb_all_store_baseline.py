import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
from typing import Union, List
from multiprocessing import Pool 
from utils import setup_logger
warnings.filterwarnings('ignore')

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

def get_data_by_store():
    
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    df = pd.concat([df, df2], axis=1)
    del df2 
    df = pd.concat([df, df3], axis=1)
    del df3 

    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]

    df = df[df['d']>=START_TRAIN].reset_index(drop=True)

    #とりあえず小さいデータセットで動くかどうかを確認する。
    #一応動いたからcolumnsを増やす(35->70くらい)
    #df = df[df['d']>=1914-365].reset_index(drop=True)
    
    return df, features

# training後にtestを再結合　
def get_base_test():
    base_test = pd.DataFrame()
#     for store_id in STORES_IDS:
#         temp_df = pd.read_pickle('test_'+store_id+'.pkl')
#         temp_df['store_id'] = store_id
#         base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    base_test = pd.read_pickle('test_'+'.pkl')
    
    return base_test


def make_lag(LAG_DAY: int):
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'sales_lag_'+str(LAG_DAY)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
    return lag_df[[col_name]]


def make_lag_roll(LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


import lightgbm as lgb
lgb_params = {
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



###以下超重要->train, test期間を調整してvalidationすることで最適な期間や、store, categoryの分割方法を見つける
VER = 1                          # modelのバージョン
SEED = 42                        
seed_everything(SEED)             
lgb_params['seed'] = SEED        
N_CORES = psutil.cpu_count()     


#LIMITS and const
TARGET      = 'sales'            
START_TRAIN = 0                  
END_TRAIN   = 1913              
P_HORIZON   = 28                 
USE_AUX     = False              

#オーバーフィットしてしまう特徴量や、testで使用できない特徴量
#この場合はstoreとstateが両方とも存在しても意味がないから
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
'''                   
mean_features   = ['enc_cat_id_mean','enc_cat_id_std',
                   'enc_dept_id_mean','enc_dept_id_std',
                   'enc_item_id_mean','enc_item_id_std'] 
'''
mean_features = ['enc_state_id_mean', 'enc_state_id_std', 'enc_store_id_mean',
                'enc_store_id_std', 'enc_cat_id_mean', 'enc_cat_id_std',
                'enc_dept_id_mean', 'enc_dept_id_std', 'enc_state_id_cat_id_mean',
                'enc_state_id_cat_id_std', 'enc_state_id_dept_id_mean',
                'enc_state_id_dept_id_std', 'enc_store_id_cat_id_mean',
                'enc_store_id_cat_id_std', 'enc_store_id_dept_id_mean',
                'enc_store_id_dept_id_std', 'enc_item_id_mean', 'enc_item_id_std',
                'enc_item_id_state_id_mean', 'enc_item_id_state_id_std',
                'enc_item_id_store_id_mean', 'enc_item_id_store_id_std']

#PATHS for Features
ORIGINAL = '../input/m5-forecasting-accuracy/'
BASE     = '../input/m5-simple-fe/grid_part_1.pkl'
PRICE    = '../input/m5-simple-fe/grid_part_2.pkl'
CALENDAR = '../input/m5-simple-fe/grid_part_3.pkl'
LAGS     = '../input/m5-lags-features/lags_df_28.pkl'
MEAN_ENC = '../input/m5-custom-features/mean_encoding_df.pkl'


# AUX(pretrained) Models paths
AUX_MODELS = '../input/m5-aux-models/'


STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())


SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])

        
grid_df, features_columns = get_data_by_store()

#preds_maskはlaf特徴量計算のため
train_mask = grid_df['d']<=END_TRAIN#1913
valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))#1913-28<x<1913
preds_mask = grid_df['d']>(END_TRAIN-100)#lag featureなどのために100日前まで取っておく。

# maskの適用とメモリ削減のためのbinary化
train_data = lgb.Dataset(grid_df[train_mask][features_columns], 
                   label=grid_df[train_mask][TARGET])
train_data.save_binary('train_data.bin')
train_data = lgb.Dataset('train_data.bin')
valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
                   label=grid_df[valid_mask][TARGET])

grid_df = grid_df[preds_mask].reset_index(drop=True)
keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]#tmpは将来的に変更する必要があるcolumns
grid_df = grid_df[keep_cols]
print(grid_df.shape)

#grid_df.to_pickle('test_'+store_id+'.pkl')
grid_df.to_pickle('test_'+'.pkl')
del grid_df



seed_everything(SEED)
estimator = lgb.train(lgb_params,
                      train_data,
                      valid_sets = [valid_data],
                      verbose_eval = 100,
                      early_stopping_rounds=10,
                      )


model_name = 'lgb_model_'+'_v'+str(VER)+'.bin'
pickle.dump(estimator, open(model_name, 'wb'))

#forループの各イテレーションでtrain_data.binを作っては削除する。
#!rm train_data.bin
os.remove('train_data.bin')
del train_data, valid_data, estimator
gc.collect()

#predictionのためにtrainに使用した特徴量を記録
MODEL_FEATURES = features_columns



all_preds = pd.DataFrame()

# log特徴量作成のためにtraintestデータを再結合
base_test = get_base_test()
print(base_test.shape)

# Timer to measure predictions time 
main_time = time.time()

#lag特徴量は計算に時間がかかるから、日毎に計算する。
#日毎、店舗ごと別々に計算している。
for PREDICT_DAY in range(1,29):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()

    # Make temporary grid to calculate rolling lags
    grid_df = base_test.copy()
    print(grid_df.shape)
    
    grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)
 
    model_path = 'lgb_model_'+'_v'+str(VER)+'.bin'
    estimator = pickle.load(open(model_path, 'rb'))
    day_mask = base_test['d']==(END_TRAIN+PREDICT_DAY)
    mask = day_mask
    base_test[TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])
    
    # Make good column naming and add 
    # to all_preds DataFrame
    temp_df = base_test[day_mask][['id',TARGET]]
    temp_df.columns = ['id','F'+str(PREDICT_DAY)]
    if 'id' in list(all_preds):
        all_preds = all_preds.merge(temp_df, on=['id'], how='left')#横に結合していくsubmissionの形状を目指すために
    else:
        all_preds = temp_df.copy()
        
    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
    del temp_df
    
all_preds = all_preds.reset_index(drop=True)
all_preds


# _evaluationを埋めるためにfillna
submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)#evaluationをfillnaする。
submission.to_csv('submission_v'+str(VER)+'.csv', index=False)