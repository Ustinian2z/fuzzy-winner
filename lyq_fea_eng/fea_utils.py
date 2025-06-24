import numpy as np
import pandas as pd
import os, json
from typing import Dict
from collections import Counter
import dask.dataframe as dd
import swifter
from joblib import Parallel, delayed
from tqdm import tqdm
from pandas.core.frame import DataFrame
from pandas.core.groupby.generic import SeriesGroupBy

base_dir = 'E:/芒果台比赛/'
options = {
    'n_jobs': 16
}

def day_id(i: int):
    return str(i) if i >= 10 else '0' + str(i)


def update_df_tmp_all(df_today: DataFrame, df_tmp: DataFrame):
    df = df_today[['did', 'vid', 'click_time', 'play_time', 'item_duration']].copy()
    df['click'] = df['click_time'].notna().astype(int)
    df['play'] = df['play_time'].fillna(0).astype(int) / df['item_duration'].astype(int)

    df = df.groupby('did', sort=False).agg(
            show_sum=('vid', 'count'),
            click_sum=('click', 'sum'),
            play_ratio_sum=('play', 'sum')
        )

    df_tmp_new = df_tmp[['did', 'show_sum', 'click_sum', 'play_ratio_sum']].copy().set_index('did')
    df_tmp_new = df_tmp_new.add(df, fill_value=0)

    return df_tmp_new

def update_df_tmp_fea(feature: str, df_today: DataFrame, df_tmp: DataFrame):
    df = df_today[['did', 'vid', 'click_time', 'play_time', 'item_duration', feature]].copy()
    df[feature] = df[feature].astype('str')
    df['click'] = df['click_time'].notna().astype(int)
    df['play'] = (df['play_time'].fillna(0).astype(int) / df['item_duration'].astype(int)).astype('float32')

    agg_df = df.groupby(['did', feature], sort=False, as_index=False).agg(
                    show_sum=('vid', 'count'),
                    click_sum=('click', 'sum'),
                    play_ratio_sum=('play', 'sum')
                )

    dict_df = zip_dic_parallel(agg_df.groupby('did'), feature)

    tmp_selected_cols = ['did', f'{feature}_show_sum', f'{feature}_click_sum', f'{feature}_play_ratio_sum']
    df_tmp_new = df_tmp[tmp_selected_cols].copy().merge(dict_df, on='did', how='left')
    for col in tmp_selected_cols[1:]:
        col_ = col[len(feature)+1:]
        df_tmp_new[col] = df_tmp_new.swifter.apply(lambda row: update_sum_dic(row[col_], row[col]), axis=1)
    df_tmp_new = df_tmp_new[tmp_selected_cols].set_index('did')

    return df_tmp_new


def update_df_all(df_tomorrow_merged: DataFrame):
    df_new = df_tomorrow_merged[['show_sum', 'click_sum', 'play_ratio_sum']].copy()
    df_new['ctr'] = (df_new['click_sum'] / df_new['show_sum']).astype('float32')
    df_new['ptr'] = (df_new['play_ratio_sum'] / df_new['click_sum']).astype('float32')
    df_new = df_new[['show_sum', 'click_sum', 'ctr', 'ptr']].fillna(0)

    return df_new

def update_df_fea(feature: str, df_tomorrow_merged: DataFrame):
    df_new = df_tomorrow_merged[[f'{feature}_show_sum', f'{feature}_click_sum', f'{feature}_play_ratio_sum']].copy()
    df_new[f'{feature}_ctr'] = df_new.swifter.apply(lambda row: get_ratio_dic(row[f'{feature}_click_sum'], row[f'{feature}_show_sum']), axis=1)
    df_new[f'{feature}_ptr'] = df_new.swifter.apply(lambda row: get_ratio_dic(row[f'{feature}_play_ratio_sum'], row[f'{feature}_click_sum']), axis=1)
    df_new = df_new[[f'{feature}_click_sum', f'{feature}_ctr', f'{feature}_ptr']]
    return df_new


def zip_dic_parallel(df_grouped: SeriesGroupBy, feature: str):
    def zip_dic(key, group, feature):
        show_dict = dict(zip(group[feature], group['show_sum']))
        click_dict = dict(zip(group[feature], group['click_sum']))
        play_dict = dict(zip(group[feature], group['play_ratio_sum']))
        return {
            'did': key,
            'show_sum': show_dict,
            'click_sum': click_dict,
            'play_ratio_sum': play_dict
        }

    results = Parallel(n_jobs = options['n_jobs'])(
        delayed(zip_dic)(key, group, feature) for key, group in df_grouped
    )
    return pd.DataFrame(results)

def update_sum_dic(new_dic: Dict[str, int], main_dic: Dict[str, int]):
    return dict(Counter(main_dic) + Counter(new_dic))

def get_ratio_dic(numerator: Dict[str, float], denominator: Dict[str, float]):
    ratio_dic = {}
    for key in numerator.keys():
        if denominator[key] != 0:
            ratio_dic[key] = numerator[key] / denominator[key]
    return ratio_dic



'''
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)
tmp_path = base_dir + f'用户历史日志_自加特征/did_attr_fea_temp.csv'
df_tmp_ = pd.read_csv(tmp_path)
input_path = base_dir + f'用户历史日志_含特征/user_history_day01.csv'
df_ = pd.read_csv(input_path)

def build_df_all(df_, df_tmp_):       # 要求df_和df_tmp_有相同的did
    df = df_[['did', 'vid', 'click_time', 'play_time', 'item_duration']].copy()
    df['click'] = df['click_time'].notna().astype(int)
    df['play'] = (df['play_time'].fillna(0).astype(int) / df['item_duration'].astype(int)).astype('float32')

    df = df.groupby('did', sort=False).agg(
                 show_sum=('vid', 'count'),
                 click_sum=('click', 'sum'),
                 play_ratio_sum=('play', 'sum')
             )

    df_tmp = df_tmp_[['did', 'show_sum', 'click_sum', 'play_ratio_sum']].copy().set_index('did')
    df_tmp = df_tmp.add(df, fill_value=0).reset_index()

    df = df_tmp.copy()
    df['ctr'] = (df['click_sum'] / df['show_sum']).astype('float32')
    df['ptr'] = (df['play_ratio_sum'] / df['click_sum']).astype('float32')
    df = df[['did', 'show_sum', 'click_sum', 'ctr', 'ptr']].fillna(0).reset_index()

    print('overall metrics processed successfully')
    return df, df_tmp
    
def build_df_fea_parallel(feature: str, df_, df_tmp_):
    df = df_[['did', 'vid', 'click_time', 'play_time', 'item_duration', feature]].copy()
    df_tmp = df_tmp_[['did', f'{feature}_show_sum', f'{feature}_click_sum', f'{feature}_play_ratio_sum']].copy()
    for col in df_tmp.columns:
        if col != 'did':
            df_tmp[col] = df_tmp[col].swifter.apply(json.loads)

    df[feature] = df[feature].astype('category')
    df['click'] = df['click_time'].notna().astype(int)
    df['play'] = (df['play_time'].fillna(0).astype(int) / df['item_duration'].astype(int)).astype('float32')

    agg_df = df.groupby(['did', feature], sort=False, as_index=False) \
             .agg(
                 show_sum=('vid', 'count'),
                 click_sum=('click', 'sum'),
                 play_ratio_sum=('play', 'sum')
             ).reset_index()

    dict_df = zip_dic_parallel(agg_df.groupby('did'), feature)
    df = df[['did']].drop_duplicates().merge(dict_df, on='did', how='left')

    df_tmp = df.merge(df_tmp, on='did', how='left').fillna(0)
    df_tmp[f'{feature}_show_sum'] = df_tmp.swifter.apply(lambda row: merge_sum_dic(row['show_sum'], row[f'{feature}_show_sum']), axis=1)
    df_tmp[f'{feature}_click_sum'] = df_tmp.swifter.apply(lambda row: merge_sum_dic(row['click_sum'], row[f'{feature}_click_sum']), axis=1)
    df_tmp[f'{feature}_play_ratio_sum'] = df_tmp.swifter.apply(lambda row: merge_sum_dic(row['play_ratio_sum'], row[f'{feature}_play_ratio_sum']), axis=1)
    df_tmp = df_tmp[['did', f'{feature}_show_sum', f'{feature}_click_sum', f'{feature}_play_ratio_sum']]

    df = df_tmp.copy()
    df[f'{feature}_ctr'] = df.swifter.apply(lambda row: get_ratio_dic_and_dump(row[f'{feature}_click_sum'], row[f'{feature}_show_sum']), axis=1)
    df[f'{feature}_ptr'] = df.swifter.apply(lambda row: get_ratio_dic_and_dump(row[f'{feature}_play_ratio_sum'], row[f'{feature}_click_sum']), axis=1)
    df = df[['did', f'{feature}_click_sum', f'{feature}_ctr', f'{feature}_ptr']].fillna('{}')

    print(f"{feature}'s metrics processed successfully")
    return df, df_tmp
'''
