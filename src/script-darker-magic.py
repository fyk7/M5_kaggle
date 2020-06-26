import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool
from utils import seed_everything, df_parallelize_run, setup_logger
import config
warnings.filterwarnings('ignore')

########################### Helper to load data by store ID
#################################################################################
def get_data_by_store(store):
    df = pd.concat([pd.read_pickle(config.BASE),
                    pd.read_pickle(config.PRICE).iloc[:,2:],
                    pd.read_pickle(config.CALENDAR).iloc[:,2:]],
                    axis=1)
    
    # Leave only relevant store
    df = df[df['store_id']==store]
    df2 = pd.read_pickle(config.MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    df3 = pd.read_pickle(config.LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2 # to not reach memory limit 
    df = pd.concat([df, df3], axis=1)
    del df3 # to not reach memory limit 
    
    features = [col for col in list(df) if col not in remove_features]

    df = df[['id','d',config.TARGET]+features]
    df = df[df['d']>=config.START_TRAIN].reset_index(drop=True)
    
    return df, features

# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in config.STORES_IDS:
        temp_df = pd.read_pickle('test_'+store_id+'.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test


########################### Helper to make dynamic rolling lags
#################################################################################
def make_lag(LAG_DAY):
    lag_df = base_test[['id','d',config.TARGET]]
    col_name = 'sales_lag_'+str(LAG_DAY)
    lag_df[col_name] = lag_df.groupby(['id'])[config.TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
    return lag_df[[col_name]]


def make_lag_roll(LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',config.TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[config.TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


########################### Model params
#################################################################################
import lightgbm as lgb
lgb_params = config.MODEL_CONFIG['LightGBM']

########################### Vars
#################################################################################
seed_everything(config.SEED)            # to be as deterministic 
lgb_params['seed'] = config.SEED        # as possible


remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',config.TARGET]

mean_features = ['enc_state_id_mean', 'enc_state_id_std', 'enc_store_id_mean',
                'enc_store_id_std', 'enc_cat_id_mean', 'enc_cat_id_std',
                'enc_dept_id_mean', 'enc_dept_id_std', 'enc_state_id_cat_id_mean',
                'enc_state_id_cat_id_std', 'enc_state_id_dept_id_mean',
                'enc_state_id_dept_id_std', 'enc_store_id_cat_id_mean',
                'enc_store_id_cat_id_std', 'enc_store_id_dept_id_mean',
                'enc_store_id_dept_id_std', 'enc_item_id_mean', 'enc_item_id_std',
                'enc_item_id_state_id_mean', 'enc_item_id_state_id_std',
                'enc_item_id_store_id_mean', 'enc_item_id_store_id_std']

#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])


########################### Train Models
#################################################################################
for store_id in config.STORES_IDS:
    print('Train', store_id)
    
    # Get grid for current store
    grid_df, features_columns = get_data_by_store(store_id)
    
    train_mask = grid_df['d']<=config.END_TRAIN
    valid_mask = train_mask&(grid_df['d']>(config.END_TRAIN-config.P_HORIZON))
    preds_mask = grid_df['d']>(config.END_TRAIN-100)
    
    train_data = lgb.Dataset(grid_df[train_mask][features_columns], 
                       label=grid_df[train_mask][config.TARGET])
    train_data.save_binary('train_data.bin')
    train_data = lgb.Dataset('train_data.bin')
    
    valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
                       label=grid_df[valid_mask][config.TARGET])
    
    # Saving part of the dataset for later predictions
    # Removing features that we need to calculate recursively 
    grid_df = grid_df[preds_mask].reset_index(drop=True)
    keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
    grid_df = grid_df[keep_cols]
    grid_df.to_pickle('test_'+store_id+'.pkl')
    del grid_df
    
    # Launch seeder again to make lgb training 100% deterministic
    # with each "code line" np.random "evolves" 
    # so we need (may want) to "reset" it
    seed_everything(config.SEED)
    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets = [valid_data],
                          verbose_eval = 100,
                          )
    
    # Save model - it's not real '.bin' but a pickle file
    model_name = 'lgb_model_'+store_id+'_v'+str(config.VER)+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))

    os.remove('train_data.bin')
    del train_data, valid_data, estimator
    gc.collect()
    
    # "Keep" models features for predictions
    MODEL_FEATURES = features_columns

########################### Predict
#################################################################################

all_preds = pd.DataFrame()

base_test = get_base_test()

main_time = time.time()

for PREDICT_DAY in range(1,29):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()

    # Make temporary grid to calculate rolling lags
    grid_df = base_test.copy()
    grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)
        
    for store_id in config.STORES_IDS:
        
        # Read all our models and make predictions
        # for each day/store pairs
        model_path = 'lgb_model_'+store_id+'_v'+str(config.VER)+'.bin' 
        estimator = pickle.load(open(model_path, 'rb'))
        
        day_mask = base_test['d']==(config.END_TRAIN+PREDICT_DAY)
        store_mask = base_test['store_id']==store_id
        
        mask = (day_mask)&(store_mask)
        base_test[config.TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])
    
    # Make good column naming and add 
    # to all_preds DataFrame
    temp_df = base_test[day_mask][['id',config.TARGET]]
    temp_df.columns = ['id','F'+str(PREDICT_DAY)]
    if 'id' in list(all_preds):
        all_preds = all_preds.merge(temp_df, on=['id'], how='left')
    else:
        all_preds = temp_df.copy()
        
    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
    del temp_df
    
all_preds = all_preds.reset_index(drop=True)
all_preds


########################### Export
#################################################################################
# we need to do fillna() for "_evaluation" items
submission = pd.read_csv(config.ORIGINAL+'sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
submission.to_csv('submission_v'+str(config.VER)+'.csv', index=False)
