#!/usr/bin/env python
# coding: utf-8

features = ['item_id', 'dept_id', 'cat_id', 'release', 'sell_price', 'price_max', 'price_min', 'price_std', 'price_mean', 
            'price_norm', 'price_nunique', 'item_nunique', 'price_momentum', 'price_momentum_m', 'price_momentum_y', 
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 
            'tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end', 
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
print(len(features))




TARGET = 'sales'
remove_features = ['id','state_id','store_id', 'date','wm_yr_wk','d',TARGET]

mean_features   = ['enc_cat_id_mean','enc_cat_id_std',  'enc_dept_id_mean', 'enc_dept_id_std', 
                                  'enc_item_id_mean','enc_item_id_std'] 

cat_cols = []

cat_id_cols = []

num_cols = []

use_cols = []






