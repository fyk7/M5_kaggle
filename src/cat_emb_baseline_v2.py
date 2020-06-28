import numpy as np
import pandas as pd
import datetime
import os, sys, gc, time, warnings, pickle, psutil, random
from tqdm import tqdm
from multiprocessing import Pool
from utils import setup_logger
rom sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

########################### logger
#################################################################################
NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
logger = setup_logger(f'./logs/train_{NOW}.log')

########################### Helpers
#################################################################################

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


########################### Helper to load data by store ID
#################################################################################
def get_data_by_store(store):
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    df = df[df['store_id']==store]
    
    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2 
    df = pd.concat([df, df3], axis=1)
    del df3 
    
    #remove_features==unuse_features
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]

    #preprocessing for cat_emb cat_cols
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].cat.codes)
    

    df = df[df['d']>=START_TRAIN].reset_index(drop=True)
    ################################3
    #print(df.info())
    
    return df, features

# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_pickle('../data/output/test_'+store_id+'.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test


########################### Helper to make dynamic rolling lags
#################################################################################
def make_lag(LAG_DAY):
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



########################### Vars
#################################################################################
VER = 1                          # Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
#lgb_params['seed'] = SEED        # as possible
N_CORES = psutil.cpu_count()     # Available CPU cores


#LIMITS and const
TARGET      = 'sales'            # Our target
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1913               # End day of our train set
P_HORIZON   = 28                 # Prediction horizon

############################## defaultの設定 ################################
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]

mean_features = ['enc_state_id_mean', 'enc_state_id_std', 'enc_store_id_mean',
                'enc_store_id_std', 'enc_cat_id_mean', 'enc_cat_id_std',
                'enc_dept_id_mean', 'enc_dept_id_std', 'enc_state_id_cat_id_mean',
                'enc_state_id_cat_id_std', 'enc_state_id_dept_id_mean',
                'enc_state_id_dept_id_std', 'enc_store_id_cat_id_mean',
                'enc_store_id_cat_id_std', 'enc_store_id_dept_id_mean',
                'enc_store_id_dept_id_std', 'enc_item_id_mean', 'enc_item_id_std',
                'enc_item_id_state_id_mean', 'enc_item_id_state_id_std',
                'enc_item_id_store_id_mean', 'enc_item_id_store_id_std']





############################## cat_embの設定 ################################
#cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
cat_id_cols = ["item_id", "dept_id", "cat_id"]
#cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1", 
#                          "event_type_1", "event_name_2", "event_type_2"]
cat_cols = ['release', 'price_nunique', 'item_nunique', 
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
            'tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end']

cat_cols = cat_id_cols + cat_cols
num_cols = ['release', 'sell_price', 'price_max', 'price_min', 'price_std', 'price_mean', 
            'price_norm', 'price_nunique', 'item_nunique', 'price_momentum', 'price_momentum_m', 'price_momentum_y', 
            #enc_ids
            'enc_cat_id_mean', 'enc_cat_id_std', 'enc_dept_id_mean', 'enc_dept_id_std', 'enc_item_id_mean', 'enc_item_id_std', 
           #sales_lag 
            'sales_lag_28', 'sales_lag_29', 'sales_lag_30', 'sales_lag_31', 'sales_lag_32', 'sales_lag_33', 'sales_lag_34', 'sales_lag_35', 
            'sales_lag_36', 'sales_lag_37', 'sales_lag_38', 'sales_lag_39', 'sales_lag_40', 'sales_lag_41', 'sales_lag_42', 
           #rolling_mean 
            'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30', 'rolling_std_30', 
            'rolling_mean_60', 'rolling_std_60', 'rolling_mean_180', 'rolling_std_180', 
           #rolling_mean_tmp 
            'rolling_mean_tmp_1_7', 'rolling_mean_tmp_1_14', 'rolling_mean_tmp_1_30', 'rolling_mean_tmp_1_60', 
            'rolling_mean_tmp_7_7', 'rolling_mean_tmp_7_14', 'rolling_mean_tmp_7_30', 
            'rolling_mean_tmp_7_60', 'rolling_mean_tmp_14_7', 'rolling_mean_tmp_14_14', 'rolling_mean_tmp_14_30', 
            'rolling_mean_tmp_14_60']
bool_cols = ['snap_CA', 'snap_TX', 'snap_WI']
dense_cols = num_cols + bool_cols




#PATHS for Features
ORIGINAL = '../data/input/m5-forecasting-accuracy/'
BASE     = '../data/input/m5-simple-fe/grid_part_1.pkl'
PRICE    = '../data/input/m5-simple-fe/grid_part_2.pkl'
CALENDAR = '../data/input/m5-simple-fe/grid_part_3.pkl'
LAGS     = '../data/input/m5-lags-features/lags_df_28.pkl'
MEAN_ENC = '../data/input/m5-custom-features/mean_encoding_df.pkl'

#STORES ids
STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())


#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])



########################### Model params
#################################################################################
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten
from tensorflow.keras.models import Model

#cat_emb訓練データ作成
#辞書型にして、catとnumのカラムをmodelに教える
#コードに問題がないことを確認したらstandard scalerを追加する。
def make_X(df):
    #cat_type_list = ['item_id','dept_id','cat_id','event_name_1','event_name_2','event_type_1','event_type_2']
    #include_minus = ['event_name_1','event_name_2','event_type_1','event_type_2']
    cat_cols = ['release', 'price_nunique', 'item_nunique', 
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
                'tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end']
    bool_type_list = ['snap_CA', 'snap_TX', 'snap_WI']
    for bl in bool_type_list:
        df[bl] = df[bl].astype(np.int8)
    X = {"dense1": df[dense_cols].to_numpy()}
    for i, v in enumerate(cat_cols):
        X[v] = df[[v]].to_numpy()
    return X

#cat_cols = ['release', 'price_nunique', 'item_nunique', 
#            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
#            'tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end']

#storeとstateは限定されているから、このモデルでは使用しない。
def create_model(lr=0.002):
    tf.keras.backend.clear_session()
    gc.collect()

    # 数値cols
    dense_input = Input(shape=(len(dense_cols), ), name='dense1')

    # Embedding input
    # cat_cols
    #wday_input = Input(shape=(1,), name='wday')
    #month_input = Input(shape=(1,), name='month')
    #year_input = Input(shape=(1,), name='year')
    release_input = Input(shape=(1,), name='release')
    price_nunique_input = Input(shape=(1,), name='price_nunique')
    item_nunique_input = Input(shape=(1,), name='item_nunique')
    tm_d_input = Input(shape=(1,), name='tm_d')
    tm_w_input = Input(shape=(1,), name='tm_w')
    tm_m_input = Input(shape=(1,), name='tm_m')
    tm_y_input = Input(shape=(1,), name='tm_y')
    tm_wm_input = Input(shape=(1,), name='tm_wm')
    tm_dw_input = Input(shape=(1,), name='tm_dw')
    tm_w_end_input = Input(shape=(1,), name='tm_w_end')

    event_name_1_input = Input(shape=(1,), name='event_name_1')
    event_type_1_input = Input(shape=(1,), name='event_type_1')
    event_name_2_input = Input(shape=(1,), name='event_name_2')
    event_type_2_input = Input(shape=(1,), name='event_type_2')
    item_id_input = Input(shape=(1,), name='item_id')
    dept_id_input = Input(shape=(1,), name='dept_id')
    #store_id_input = Input(shape=(1,), name='store_id')
    cat_id_input = Input(shape=(1,), name='cat_id')
    #state_id_input = Input(shape=(1,), name='state_id')

    #wday_emb = Flatten()(Embedding(7, 1)(wday_input))
    #month_emb = Flatten()(Embedding(12, 1)(month_input))
    #year_emb = Flatten()(Embedding(6, 1)(year_input))
    release_emb = Flatten()(Embedding(259,3)(release_input))
    price_nunique_emb = Flatten()(Embedding(20,1)(price_nunique_input))
    item_nunique_emb = Flatten()(Embedding(184, 3)(item_nunique_input))
    tm_d_emb = Flatten()(Embedding(31,1)(tm_d_input))
    tm_w_emb = Flatten()(Embedding(53,1)(tm_w_input))
    tm_m_emb = Flatten()(Embedding(12,1)(tm_m_input))
    tm_y_emb = Flatten()(Embedding(6,1)(tm_y_input))
    tm_wm_emb = Flatten()(Embedding(5,1)(tm_wm_input))
    tm_dw_emb = Flatten()(Embedding(7,1)(tm_dw_input))
    tm_w_end_emb = Flatten()(Embedding(2,1)(tm_w_end_input))
    '''
    release_emb = Flatten()(Embedding(503,3)(release_input))
    price_nunique_emb = Flatten()(Embedding(22,1)(price_nunique_input))
    item_nunique_emb = Flatten()(Embedding(306, 3)(item_nunique_input))
    tm_d_emb = Flatten()(Embedding(32,1)(tm_d_input))
    tm_w_emb = Flatten()(Embedding(54,1)(tm_w_input))
    tm_m_emb = Flatten()(Embedding(13,1)(tm_m_input))
    tm_y_emb = Flatten()(Embedding(6,1)(tm_y_input))
    tm_wm_emb = Flatten()(Embedding(6,1)(tm_wm_input))
    tm_dw_emb = Flatten()(Embedding(7,1)(tm_dw_input))
    tm_w_end_emb = Flatten()(Embedding(2,1)(tm_w_end_input))
    '''

    '''nunique()の結果は以下のようになった。もし問題があれば以下の値に置換する
    event_name_1       30
    event_type_1        4
    event_name_2        4
    event_type_2        2
    '''

    event_name_1_emb = Flatten()(Embedding(31, 1)(event_name_1_input))
    event_type_1_emb = Flatten()(Embedding(5, 1)(event_type_1_input))
    event_name_2_emb = Flatten()(Embedding(5, 1)(event_name_2_input))
    event_type_2_emb = Flatten()(Embedding(5, 1)(event_type_2_input))

    item_id_emb = Flatten()(Embedding(3049, 3)(item_id_input))
    dept_id_emb = Flatten()(Embedding(7, 1)(dept_id_input))
    #store_id_emb = Flatten()(Embedding(10, 1)(store_id_input))
    cat_id_emb = Flatten()(Embedding(3, 1)(cat_id_input))
    #state_id_emb = Flatten()(Embedding(3, 1)(state_id_input))

    # Combine dense and embedding parts and add dense layers. Exit on linear scale.
    '''
    x = concatenate([dense_input, wday_emb, month_emb, year_emb, 
                     event_name_1_emb, event_type_1_emb, 
                     event_name_2_emb, event_type_2_emb, 
                     item_id_emb, dept_id_emb, cat_id_emb,
                     store_id_emb, state_id_emb])
    '''
    x = concatenate([dense_input,
                     release_emb, price_nunique_emb, item_nunique_emb,
                     tm_d_emb, tm_w_emb, tm_m_emb, tm_y_emb, tm_wm_emb, tm_dw_emb, tm_w_end_emb,
                     event_name_1_emb, event_type_1_emb, 
                     event_name_2_emb, event_type_2_emb, 
                     item_id_emb, dept_id_emb, cat_id_emb])
    
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(4, activation="relu")(x)

    outputs = Dense(1, activation="linear", name='output')(x)

    #column名とそのcolumnが入るInput layer
    '''
    inputs = {"dense1": dense_input, "wday": wday_input, "month": month_input, "year": year_input, 
              "event_name_1": event_name_1_input, "event_type_1": event_type_1_input,
              "event_name_2": event_name_2_input, "event_type_2": event_type_2_input,
              "item_id": item_id_input, "dept_id": dept_id_input, "store_id": store_id_input, 
              "cat_id": cat_id_input, "state_id": state_id_input}
    '''
    #おそらく以下のdictでcolumn明からInputを判断する。
    inputs = {"dense1": dense_input, 
              "release": release_input, "price_nunique": price_nunique_input, "item_nunique": item_nunique_input,
              "tm_d": tm_d_input, "tm_w": tm_w_input, "tm_m": tm_m_input, "tm_y": tm_y_input,
              "tm_wm": tm_wm_input, "tm_dw": tm_dw_input, "tm_w_end": tm_w_end_input,
              "event_name_1": event_name_1_input, "event_type_1": event_type_1_input,
              "event_name_2": event_name_2_input, "event_type_2": event_type_2_input,
              "item_id": item_id_input, "dept_id": dept_id_input, "cat_id": cat_id_input}
    # Connect input and output
    model = Model(inputs, outputs)

    model.compile(loss=keras.losses.mean_squared_error,
                  metrics=["mse"],
                  optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


########################### Train Models
#################################################################################
for store_id in STORES_IDS:
    logger.info(f'start train {store_id}')
    
    # Get grid for current store
    grid_df, features_columns = get_data_by_store(store_id)
    logger.info(f'grid_df shape: {grid_df.shape}')
    logger.info(f'features: {features_columns}')

    # cat_emb用のfillna処理
    for i, v in tqdm(enumerate(num_cols)):
        grid_df[v] = grid_df[v].fillna(grid_df[v].median())
    logger.info(f'num_cols: {num_cols}')
    
    train_mask = grid_df['d']<=END_TRAIN
    valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))
    preds_mask = grid_df['d']>(END_TRAIN-100)
    
    '''
    train_data = lgb.Dataset(grid_df[train_mask][features_columns], 
                       label=grid_df[train_mask][TARGET])
    train_data.save_binary('train_data.bin')
    train_data = lgb.Dataset('train_data.bin')
    
    valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
                       label=grid_df[valid_mask][TARGET])
    '''

    logger.info('start make datasets (X_train, y_train, valid)')

    X_train = make_X(grid_df[train_mask][features_columns])
    y_train = np.asarray(grid_df[train_mask][TARGET])#.to_numpy()
    valid = (make_X(grid_df[valid_mask][features_columns]), np.asarray(grid_df[valid_mask][TARGET]))

    logger.info(f'X_train shape: {X_train.keys()}')
    logger.info(f'y_train shape: {y_train.shape}')

    # Saving part of the dataset for later predictions
    # Removing features that we need to calculate recursively 
    grid_df = grid_df[preds_mask].reset_index(drop=True)
    keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
    grid_df = grid_df[keep_cols]
    grid_df.to_pickle('../data/output/test_'+store_id+'.pkl')

    logger.info(f'keep_cols: {keep_cols}')    
    logger.info(f'grid_df[preds_mask]: {grid_df.shape}')
    del grid_df
    
    seed_everything(SEED)
    '''
    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets = [valid_data],
                          verbose_eval = 100,
                          )
    
    model_name = 'lgb_model_'+store_id+'_v'+str(VER)+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))
    '''
    estimator = create_model(lr=0.002)
    estimator.summary()
    logger.info(estimator.summary())

    #X_trainは辞書
    history = estimator.fit(X_train, 
                    y_train,
                    batch_size=2 ** 14,
                    epochs=70,
                    #shuffleはどうするのか?
                    shuffle=True,
                    validation_data=valid)

    model_name = '../data/output/cat_emb_model_'+store_id+'_v'+str(VER)+'.h5'
    estimator.save(model_name)

    #!rm train_data.bin
    #os.remove('train_data.bin')
    #del train_data, valid_data, estimator
    del X_train, y_train, estimator
    gc.collect()
    
    # "Keep" models features for predictions
    MODEL_FEATURES = features_columns
    logger.info(f'keep features without _tmp_ : {MODEL_FEATURES}')

########################### Predict
#################################################################################

all_preds = pd.DataFrame()
base_test = get_base_test()
main_time = time.time()

for PREDICT_DAY in range(1,29):    
    logger.info(f'Predict | Day: {PREDICT_DAY}')
    start_time = time.time()

    # Make temporary grid to calculate rolling lags
    logger.info('start remake grid_df (for lag_roll featuers)')
    grid_df = base_test.copy()
    grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)
    #standard scalarを加えるならここに。scalar.fit(grid_df[num_cols])
    logger.info(f'test grid_df shape: {grid_df.shape}')
        
    logger.info(f'start predict for each store')
    for store_id in STORES_IDS:
        '''
        model_path = 'lgb_model_'+store_id+'_v'+str(VER)+'.bin' 
        estimator = pickle.load(open(model_path, 'rb'))
        '''
        model_path = '../data/output/cat_emb_model_'+store_id+'_v'+str(VER)+'.h5' 
        estimator = keras.models.load_model(model_path)

        day_mask = base_test['d']==(END_TRAIN+PREDICT_DAY)
        store_mask = base_test['store_id']==store_id
        
        mask = (day_mask)&(store_mask)
        X_test = make_X(grid_df[mask][MODEL_FEATURES])
        base_test[TARGET][mask] = estimator.predict(X_test)
    
    # Make good column naming and add 
    # to all_preds DataFrame
    temp_df = base_test[day_mask][['id',TARGET]]
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
submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
submission.to_csv('../data/output/submission_v'+str(VER)+'.csv', index=False)
