from typing import List
import json
import numpy as np
import pandas as pd
import pyarrow
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import fea_utils
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

base_dir = 'E:/芒果台比赛/'
item_fea = ["item_cid", "item_type", "item_assetSource", "item_classify", "item_isIntact", "sid", "stype"]


def merge_history_seq():
    for i in range(1, 30+1):
        day_id =  fea_utils.day_id(i)
        df_play = pd.read_csv(base_dir + '用户历史播放数据/' + f'day{day_id}/day{day_id}_data.csv', usecols=['did','vid','play_time'])
        df_click = pd.read_csv(base_dir + '用户历史点击数据/' + f'day{day_id}/day{day_id}_data.csv')
        df_show = pd.read_csv(base_dir + '用户历史曝光数据/' + f'day{day_id}/did_show_data{day_id}.csv')

        df_show.merge(df_click, on=['did','vid'], how='outer') \
               .merge(df_play, on=['did','vid'], how='outer') \
               .sort_values(by=['did', 'vid']).drop_duplicates(subset=['did', 'vid']) \
               .to_csv(base_dir + '用户历史日志/' + f'user_history_day{day_id}.csv', index=False)
        print(f"day{day_id} processed successfully")


def merge_item_fea_multithreaded():
    item_fea = pd.read_csv(base_dir + 'vid_info/vid_info_table.csv')

    def merge_item_fea(i: int):
        day_id = fea_utils.day_id(i)
        input_path = base_dir + f'用户历史日志/user_history_day{day_id}.csv'
        output_path = base_dir + f'用户历史日志_含特征/user_history_day{day_id}.csv'

        df = pd.read_csv(input_path)
        df = df.merge(item_fea, on='vid', how='left') \
            .drop(columns=['item_cid_x']).rename(columns={'item_cid_y': 'item_cid'})
        df.to_csv(output_path, index=False)
        return day_id

    futures = []
    with ThreadPoolExecutor(max_workers=None) as executor:
        for i in range(1, 1+30):
            futures.append(executor.submit(merge_item_fea, i))

        for future in as_completed(futures):
            day_id = future.result()
            print(f"day{day_id} processed successfully")


#merge_history_seq()
#merge_item_fea_multithreaded()


def comp_test_and_train():
    df_test = pd.read_csv(base_dir + f'A榜用户曝光数据/testA_did_show.csv')
    set_test_did = set(df_test['did'].tolist())
    set_test_vid = set(df_test['vid'].tolist())
    set_train_did = set(pd.read_csv(base_dir + f'did_features/did_features_table.csv')['did'].tolist())
    set_train_vid = set(pd.read_csv(base_dir + f'vid_info/vid_info_table.csv')['vid'].tolist())
    did_intersect = set_test_did & set_train_did
    vid_intersect = set_test_vid & set_train_vid
    print(f"未在训练集中现过的did在测试集中占比：{1 - len(did_intersect)/len(set_test_did)}\n"
          f"未在训练集中现过的vid在测试集中占比：{1 - len(vid_intersect)/len(set_test_vid)}")
#comp_test_and_train()


def build_init_attr_fea(cols: List[str]):
    df = pd.read_csv(base_dir + 'did_features/did_features_table.csv', usecols=['did'])
    df["show_sum"] = 0; df["click_sum"] = 0; df["play_ratio_sum"] = 0
    for col in cols:
        df[f"{col}_show_sum"] = '{}'; df[f"{col}_click_sum"] = '{}'; df[f"{col}_play_ratio_sum"] = '{}'
    for col in df.columns[4:]:
        df[col] = df[col].swifter.apply(json.loads)
    print("initiated successfully")
    return df


def build_fea_multithreaded(fea_cols: List[str], start_idx: int = 1, end_idx: int = 30):
    df_tmp = build_init_attr_fea(fea_cols)

    fea_cols = ['all'] + fea_cols

    def update_df_tmp_bind(fea: str):
        if fea == 'all':
            df_tmp_new = fea_utils.update_df_tmp_all(df_today, df_tmp_to_update)
        else:
            df_tmp_new = fea_utils.update_df_tmp_fea(fea, df_today, df_tmp_to_update)
        return df_tmp_new

    def update_df_bind(fea: str):
        if fea == 'all':
            df_new = fea_utils.update_df_all(df_tomorrow_merged)
        else:
            df_new = fea_utils.update_df_fea(fea, df_tomorrow_merged)
        return df_new


    for i in range(start_idx, end_idx):
        today_id, tomorrow_id = fea_utils.day_id(i), fea_utils.day_id(i+1)
        today_path = base_dir + f'用户历史日志_含特征/user_history_day{today_id}.csv'
        tomorrow_path = base_dir + f'用户历史日志_含特征/user_history_day{tomorrow_id}.csv'
        output_path = base_dir + f'用户历史日志_自加特征/user_history_day{tomorrow_id}.parquet'
        df_today = pd.read_csv(today_path) if i == start_idx else df_tomorrow
        df_tomorrow = pd.read_csv(tomorrow_path).set_index('did')

        # 筛选今天有交互行为的did，利用df_today中的交互信息更新df_tmp_to_update，进而得到df_tmp_new
        filter_row = df_tmp['did'].isin(df_today['did'])
        df_tmp_unchanged, df_tmp_to_update = df_tmp[~filter_row], df_tmp[filter_row]
        # 最后合并得到全新的df_tmp
        df_tmp_new = df_tmp_to_update[['did']].set_index('did')
        for col in fea_cols:
            df_tmp_new = df_tmp_new.merge(update_df_tmp_bind(col), how='left', left_index=True, right_index=True)
            print(f"{col}_metrics updated on {today_id}'s df_tmp successfully")
        df_tmp_new.reset_index(inplace=True)
        df_tmp = pd.concat([df_tmp_new, df_tmp_unchanged], ignore_index=True)
        del df_tmp_new, df_tmp_unchanged, df_tmp_to_update

        # 筛选明天的unique did，并拼接上df_tmp的信息，进而计算cnt,ctr,ptr等特征
        df_tomorrow_merged = pd.DataFrame(df_tomorrow.index.drop_duplicates()).merge(df_tmp, on='did', how='left').set_index('did')
        for col in fea_cols:
            df_tomorrow = df_tomorrow.merge(update_df_bind(col), how='left', left_index=True, right_index=True)
            print(f"{col}_metrics merged into {tomorrow_id}'s df successfully")
        df_tomorrow.reset_index(inplace=True)
        for col in df_tomorrow.columns:
            if isinstance(df_tomorrow.loc[0,col], dict):
                df_tomorrow[col] = df_tomorrow[col].swifter.apply(json.dumps)
        df_tomorrow.to_parquet(output_path, compression='snappy', index=False)

        print("---"*20 + f"day{today_id}'s over"+ "---"*20)



#build_fea_multithreaded(item_fea, 1, 30)


def compress_bullet_chat_file():
    df = pd.DataFrame({'vid': [], 'content': [], 'ctime': [], 'etime': []})
    for i in range(1, 1 + 20):
        df_tmp = pd.read_excel(base_dir + f'弹幕文本数据/{i}.xlsx', sheet_name='Sheet1',
                               usecols=['videoid', 'content', 'ctime', 'etime']).rename(columns={"videoid": "vid"})
        df = pd.concat([df, df_tmp], axis=0)
        print(i)
    for col in df.columns:
        df[col] = df[col].astype(int) if col != 'content' else df['content'].astype(str).fillna("")
    df.to_parquet(base_dir + f'弹幕文本数据/bullets.parquet')

def analyze_bullet_chat_emotion():
    df = pd.read_parquet(base_dir + '弹幕文本数据/bullets.parquet')
    vids = pd.read_csv(base_dir + 'vid_info/vid_info_table.csv', usecols=['vid'])
    df = df[df['vid'].isin(vids['vid'])]\
        .groupby('vid', as_index=False)\
        .agg(content=('content', '。'.join),
             bullet_cnt=('content', 'size'))
    df['bullet_len'] = df['content'].apply(lambda x: len(x))
    df['model_res'] = None
    classifier = pipeline(Tasks.text_classification, 'iic/nlp_structbert_sentiment-classification_chinese-base')

    def get_emotion(index, row):
        res = classifier(input=row['content'])
        return index, dict(zip(res['labels'], res['scores']))

    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(get_emotion, index, row) for index, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Rows"):
            index, dic = future.result()
            df.loc[index, 'model_res'] = dic

    df.to_parquet(base_dir + '弹幕文本数据/bullet_fea.parquet')
    df.to_csv(base_dir + '弹幕文本数据/bullet_fea.csv', index=False, encoding='utf_8_sig')


#analyze_bullet_chat_emotion()




