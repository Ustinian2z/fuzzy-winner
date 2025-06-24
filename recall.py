import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import os
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 数据路径
DATA_PATH = "./"
CLICK_PATH = os.path.join(DATA_PATH, "用户历史点击数据")
PLAY_PATH = os.path.join(DATA_PATH, "用户历史播放数据")
SHOW_PATH = os.path.join(DATA_PATH, "用户历史曝光数据")
TEST_PRED_PATH = os.path.join(DATA_PATH, "./A榜待预测的did/testA_pred_did.csv")
TEST_SHOW_PATH = os.path.join(DATA_PATH, "./A榜用户曝光数据/testA_did_show.csv")
VID_INFO_PATH = os.path.join(DATA_PATH, "./vid_info/vid_info_table.csv")
DID_FEATURES_PATH = os.path.join(DATA_PATH, "did_features/did_features_table.csv")
DANMU_PATH = os.path.join(DATA_PATH, "弹幕文本数据")

# 内存优化参数
CHUNK_SIZE = 500000
PARALLEL_JOBS = max(1, cpu_count() - 2)

# 分块加载数据
def load_csv_chunks(file_path, day, dtype=None):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=dtype):
        chunk['day'] = day 
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

# 加载曝光点击播放数据
def load_day_data(data_type, day):
    day_str = f"{day:02d}"
    path = {
        "click": os.path.join(CLICK_PATH, f"day{day_str}/day{day_str}_data.csv"),
        "play": os.path.join(PLAY_PATH, f"day{day_str}/day{day_str}_data.csv"),
        "show": os.path.join(SHOW_PATH, f"day{day_str}/did_show_data{day_str}.csv")
    }.get(data_type)
    
    return load_csv_chunks(path, day)


# 读取弹幕数据文件
def read_danmu_files():
    all_dfs = []
    for i in tqdm(range(1, 19), desc="读取弹幕数据文件"): 
        file_name = f"{i}.xlsx"
        file_path = os.path.join(DANMU_PATH, file_name)
        df = pd.read_excel(file_path)
        all_dfs.append(df)
    print("弹幕数据读取完成...")
    result = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    return result


# 加载弹幕数据
def load_danmu_data():
    df_danmu = read_danmu_files()
    if not df_danmu.empty:  
        df_danmu = df_danmu.rename(columns={'videoid': 'vid'})
        print(f"弹幕数据加载完成，记录总数: {len(df_danmu)}")
    else:
        print("未加载到弹幕数据")
    return df_danmu


# 计算视频弹幕热度
def calculate_danmu_popularity(df_danmu):
    print("计算视频弹幕热度")
    if df_danmu.empty:
        print("弹幕数据为空...")
        return {}
    
    # 按视频统计弹幕数量
    vid_danmu_counts = df_danmu['vid'].value_counts().to_dict()
    
    # 计算弹幕密度（弹幕数量/视频时长）
    df_vid_info = pd.read_csv("./vid_info/vid_info_table.csv")
    vid_to_duration = df_vid_info.set_index('vid')['item_duration'].to_dict()

    vid_density = {}
    for vid, counts in vid_danmu_counts.items():
        duration = vid_to_duration.get(vid, 1000)
        vid_density[vid] = counts / duration
    
    danmu_features = {
        'vid_danmu_count': vid_danmu_counts,
        'vid_danmu_density': vid_density
    }
    
    return danmu_features


# 加载数据
def load_full_data():    
    # 加载点击及播放数据
    print("\n加载点击数据...")
    click_data = []
    for day in tqdm(range(1, 31), desc="天"):
        click = load_day_data("click", day)
        play = load_day_data("play", day)
        
        if not click.empty and not play.empty:
            click = click.merge(play[['did', 'vid', 'play_time']], on=['did', 'vid'], how='left')
        click_data.append(click)
    
    df_click = pd.concat(click_data, ignore_index=True)
    print(f"点击数据加载完成，点击记录总数: {len(df_click)}")
    
    # 加载曝光数据
    print("\n加载曝光数据...")
    show_data = []
    for day in tqdm(range(1, 31), desc="天"):
        show = load_day_data("show", day)
        show_data.append(show)
    
    df_show = pd.concat(show_data, ignore_index=True)
    print(f"曝光数据加载完成，曝光记录总数: {len(df_show)}")
    
    # 加载视频信息
    print("\n加载视频信息...")
    df_vid_info = pd.read_csv(VID_INFO_PATH)
    
    # 合并视频合集信息
    vid_to_item_cid = df_vid_info.set_index('vid')['item_cid'].to_dict()
    df_click['item_cid'] = df_click['vid'].map(vid_to_item_cid).fillna(-1)
    df_show['item_cid'] = df_show['vid'].map(vid_to_item_cid).fillna(-1)
    
    # 加载弹幕数据
    df_danmu = load_danmu_data()
    danmu_features = calculate_danmu_popularity(df_danmu)
    
    return df_click, df_show, df_vid_info, df_danmu, danmu_features

# 加载用户特征
def load_did_features():
    print("\n加载用户特征...")
    feature_cols = [f"f{i}" for i in range(88)]
    
    # 定义数据类型
    dtype_dict = {col: np.float32 for col in feature_cols if col != 'f87'}
    dtype_dict['f87'] = 'object'  
    dtype_dict['did'] = 'object'
    
    usecols = ['did'] + feature_cols
    df_did = pd.read_csv(DID_FEATURES_PATH, usecols=usecols, dtype=dtype_dict)
    
    # 处理缺失值
    for col in feature_cols:
        if col in df_did.columns:
            if col == 'f87':
                df_did[col] = df_did[col].fillna('null')
            else:
                df_did[col] = df_did[col].fillna(0)
    
    return df_did.set_index('did'), feature_cols

# 特征工程
def build_features(df_click, df_show, df_danmu, danmu_features, start_date=21, end_date=30):
    print("\n===== 构建特征 =====")
    
    df_click_filtered = df_click[(df_click['day'] >= start_date) & (df_click['day'] < end_date)]
    df_show_filtered = df_show[(df_show['day'] >= start_date) & (df_show['day'] < end_date)]
    
    data = {}
    
    # 用户点击特征
    data['did_click_unique_vid'] = df_click_filtered.groupby('did')['vid'].nunique().to_dict()
    data['did_click_unique_item_cid'] = df_click_filtered.groupby('did')['item_cid'].nunique().to_dict()
    
    # 视频点击特征
    data['vid_click_count'] = df_click_filtered['vid'].value_counts().to_dict()
    data['item_cid_click_count'] = df_click_filtered['item_cid'].value_counts().to_dict()
    
    # 交叉点击特征
    data['did_vid_clicks'] = defaultdict(int)
    for (did, vid), count in df_click_filtered.groupby(['did', 'vid']).size().items():
        data['did_vid_clicks'][(did, vid)] = count
    
    # 计算CTR特征
    print("计算视频CTR...")
    df_all = pd.concat([
        df_click[['did', 'vid', 'item_cid']].assign(action='click'),
        df_show[['did', 'vid', 'item_cid']].assign(action='show')
    ], ignore_index=True)
    
    # 分块计算CTR
    vid_stats = defaultdict(lambda: {'clicks': 0, 'impressions': 0})
    for i in range(0, len(df_all), CHUNK_SIZE):
        chunk = df_all.iloc[i:i+CHUNK_SIZE]
        for vid, group in chunk.groupby('vid'):
            clicks = (group['action'] == 'click').sum()
            impressions = len(group)
            vid_stats[vid]['clicks'] += clicks
            vid_stats[vid]['impressions'] += impressions
    
    # 转换为CTR字典
    data['vid_ctr'] = {
        vid: stats['clicks'] / stats['impressions'] if stats['impressions'] > 0 else 0
        for vid, stats in vid_stats.items()
    }
    
    # 释放内存
    del df_all, vid_stats
    
    return data

# 构建播放率预测模型特征
def build_play_rate_features(df_click, df_show, df_vid_info, df_did_features, df_danmu, danmu_features, start_day=21, end_day=30):
    print("\n===== 构建播放率预测模型特征 =====")
    
    # 筛选时间段内的数据
    df_click_filtered = df_click[(df_click['day'] >= start_day) & (df_click['day'] < end_day)].copy()
    df_show_filtered = df_show[(df_show['day'] >= start_day) & (df_show['day'] < end_day)].copy()
    
    # 标记播放行为
    df_click_filtered['played'] = 1
    df_show_filtered['played'] = 0
    
    # 合并点击和曝光数据
    print("合并点击和曝光数据...")
    df_all = pd.concat([
        df_click_filtered[['did', 'vid', 'played']],
        df_show_filtered[['did', 'vid', 'played']]
    ], ignore_index=True)
    
    # 去重
    df_all = df_all.sort_values('played', ascending=False).drop_duplicates(subset=['did', 'vid'])
    
    # 合并用户特征
    print("合并用户特征...")
    df_all = df_all.merge(df_did_features, left_on='did', right_index=True, how='left')
    
    # 合并视频特征
    print("合并视频特征...")
    df_all = df_all.merge(df_vid_info, on='vid', how='left')
    
    
    # 处理缺失值
    print("处理缺失值...")
    df_all.fillna(0, inplace=True)
    
    # 获取用户特征
    did_feature_cols = [col for col in df_did_features.columns if col.startswith('f')]
    video_feature_cols = ['item_type', 'item_duration', 'item_assetSource', 'item_classify', 'item_isIntact']
    
    # 加入弹幕特征列
    danmu_feature_cols = ['vid_danmu_count', 'vid_danmu_density']
    all_feature_cols = did_feature_cols + video_feature_cols + danmu_feature_cols
    
    # 分类特征处理
    print("分类特征处理...")
    categorical_cols = []
    
    for col in did_feature_cols:
        if df_all[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_all[col]):
            categorical_cols.append(col)
            print(f"用户分类特征: {col}")
    

    encoders = {}
    
    # 对分类特征进行编码（使用LabelEncoder）
    print("对分类特征进行编码...")
    for col in categorical_cols:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str))
        encoders[col] = le 
    
    # 构建完整特征列表
    valid_feature_cols = []
    for col in all_feature_cols:
        if col in df_all.columns:
            valid_feature_cols.append(col)
        else:
            print(f"特征 {col} 不存在于数据中，将被忽略")
    
    # 优化数据类型以减少内存占用
    for col in valid_feature_cols:
        if df_all[col].dtype == 'float64':
            df_all[col] = df_all[col].astype(np.float32)
        elif df_all[col].dtype == 'int64':
            df_all[col] = df_all[col].astype(np.int32)
    
    print(f"使用的特征列: {valid_feature_cols}")
    
    return df_all, valid_feature_cols, encoders

# 训练LightGBM播放率预测模型
def train_play_rate_model(df_all, feature_cols, target_col='played'):
    print("\n===== 训练播放率预测模型 =====")
    
    # 划分训练集和验证集
    X = df_all[feature_cols]
    y = df_all[target_col]
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # 设置参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 45,         
        'learning_rate': 0.05,
        'feature_fraction': 0.8,  
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42,
        'is_unbalance': True 
    }
    
    # 训练模型
    print("开始训练模型...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,    
        valid_sets=[valid_data],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # 评估模型
    y_pred = model.predict(X_valid)
    auc = roc_auc_score(y_valid, y_pred)
    print(f"验证集AUC: {auc:.4f}")
    
    # 输出特征重要性
    print("\n特征重要性:")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importance()
    }).sort_values('Importance', ascending=False)
    print(feature_importance.head(20))
    
    # 释放内存
    del X, y, X_train, X_valid, y_train, y_valid, train_data, valid_data
    
    return model

# 预测播放率分数
def predict_play_rate(model, df_test, feature_cols, encoders):
    print("\n===== 预测播放率分数 =====")
    
    # 确保测试集包含所有特征列
    missing_cols = set(feature_cols) - set(df_test.columns)
    for col in missing_cols:
        df_test[col] = 0
    
    # 对分类特征应用相同的编码器
    print("分类特征编码...")
    for col, encoder in encoders.items():
        if col in df_test.columns:
            df_test[col] = df_test[col].astype(str)
            df_test[col] = df_test[col].map(
                {val: i for i, val in enumerate(encoder.classes_)}
            ).fillna(-1).astype(int)
        else:
            print(f"特征 {col} 不在测试数据中")
    
    # 分块预测以减少内存压力
    print("分块预测分数...")
    df_test['play_rate_score'] = 0.0
    
    for i in range(0, len(df_test), CHUNK_SIZE):
        chunk = df_test.iloc[i:i+CHUNK_SIZE]
        df_test.loc[i:i+CHUNK_SIZE-1, 'play_rate_score'] = model.predict(chunk[feature_cols])
    
    # 构建(did, vid)到分数的映射
    print("构建分数映射字典...")
    play_rate_dict = defaultdict(float)
    
    for i in range(0, len(df_test), CHUNK_SIZE):
        chunk = df_test.iloc[i:i+CHUNK_SIZE]
        for _, row in chunk.iterrows():
            play_rate_dict[(row['did'], row['vid'])] = row['play_rate_score']
    
    # 释放内存
    del df_test
    
    return play_rate_dict

# 计算加权分数
def calculate_score(vid, did, recall_strategy, weights, user_click_dict, vid_ctr, item_cid_popular, global_hot_rank, play_rate_dict=None, danmu_features=None):
    score = 0.0
    
    if recall_strategy == 'history_click':
        click_day = user_click_dict[did].get(vid, 0)
        score = weights['history_click'] * click_day / 30  # 越近点击分数越高
    
    elif recall_strategy == 'ctr':
        score = weights['ctr'] * vid_ctr.get(vid, 0.0)
    
    elif recall_strategy == 'item_cid_popular':
        item_cid = item_cid_popular.get(vid, [])
        score = weights['item_cid_popular'] * (item_cid.index(vid) + 1) if vid in item_cid else 0
    
    elif recall_strategy == 'global_hot':  # 可优化
        global_rank = global_hot_rank.get(vid, np.inf)
        score = weights['global_hot']
    
    elif recall_strategy == 'play_rate_model' and play_rate_dict:
        score = weights['play_rate_model'] * play_rate_dict.get((did, vid), 0.0)
      

    return score

# 召回处理函数
def process_batch(args):
    user_exposure_dict, user_exposure_counts, user_click_dict, vid_ctr, item_cid_popular, global_hot_rank, weights, df_vid_info, batch_dids, play_rate_dict, user_history_vids, danmu_features = args
    batch_results = {}
    
    # 视频合集映射
    vid_to_item_cid = df_vid_info.set_index('vid')['item_cid'].to_dict()
    
    for did in batch_dids:
        # 获取用户历史播放过的视频
        history_vids = user_history_vids.get(did, set())
        
        # 从曝光候选集中剔除历史播放过的视频
        candidates = set(user_exposure_dict.get(did, [])) - history_vids
        if not candidates:
            continue
        
        # 判断用户是否有历史点击记录
        has_history = bool(user_click_dict.get(did))
        
        # 初始化各策略召回结果
        recall_results = {
            'history_click': set(user_click_dict.get(did, {}).keys()) & candidates if has_history else set(),
            'ctr': set(),
            'item_cid_popular': set(),
            'global_hot': set(global_hot_rank.keys()) & candidates,
            'play_rate_model': {vid for vid in candidates if (did, vid) in play_rate_dict} if not has_history else set()
        }
        
        # CTR召回：基于用户历史点击的视频类别
        if has_history:
            for vid in user_click_dict[did]:
                item_cid = vid_to_item_cid.get(vid, -1)
                if item_cid != -1 and item_cid in item_cid_popular:
                    top_ctr_vids = item_cid_popular[item_cid]
                    for top_vid in top_ctr_vids:
                        if top_vid in candidates:
                            recall_results['ctr'].add(top_vid)
                            break
        
        # 同类热门召回
        for item_cid in set(vid_to_item_cid.values()):
            if item_cid != -1 and item_cid in item_cid_popular:
                popular_vids_set = set(item_cid_popular[item_cid])
                recall_results['item_cid_popular'].update(popular_vids_set & candidates)
        
        
        # 计算所有候选vid的加权分数
        all_candidates = set.union(*recall_results.values())
        if not all_candidates:
            continue
        
        best_vid = None
        max_score = -np.inf
        
        for vid in all_candidates:
            history_score = calculate_score(vid, did, 'history_click', weights, user_click_dict, vid_ctr, item_cid_popular, global_hot_rank, play_rate_dict)
            ctr_score = calculate_score(vid, did, 'ctr', weights, user_click_dict, vid_ctr, item_cid_popular, global_hot_rank, play_rate_dict)
            popular_score = calculate_score(vid, did, 'item_cid_popular', weights, user_click_dict, vid_ctr, item_cid_popular, global_hot_rank, play_rate_dict)
            global_hot_score = calculate_score(vid, did, 'global_hot', weights, user_click_dict, vid_ctr, item_cid_popular, global_hot_rank, play_rate_dict)
            play_rate_score = calculate_score(vid, did, 'play_rate_model', weights, user_click_dict, vid_ctr, item_cid_popular, global_hot_rank, play_rate_dict)

            total_score = (history_score + ctr_score + popular_score + 
                          global_hot_score + play_rate_score)
            
            if total_score > max_score:
                max_score = total_score
                best_vid = vid
        
        if best_vid:
            batch_results[did] = best_vid
            continue

        # 兜底策略：曝光次数最多的vid
        if did in user_exposure_counts:
            sorted_vids = sorted(
                user_exposure_counts[did].items(), 
                key=lambda x: (-x[1], x[0]) 
            )
            for vid, _ in sorted_vids:
                if vid in candidates:
                    batch_results[did] = vid
                    break
            if did in batch_results:
                continue

        # 兜底策略：随机选择曝光vid
        if candidates:
            batch_results[did] = np.random.choice(list(candidates))
    
    return batch_results

# 多策略召回主函数
def multi_strategy_recall(df_click, df_show, df_vid_info, df_did_features, test_dids, test_show_path, recent_days=16, weights=None, custom_show_df=None, danmu_features=None):
    print("\n===== 开始多策略召回 =====")
    start_time = time.time()
    cutoff_day = 30 - recent_days
    recent_click = df_click[df_click['day'] >= cutoff_day].copy()

    # 处理曝光数据
    if custom_show_df is not None:  # 用于验证集
        test_show_dedup = custom_show_df.drop_duplicates(subset=['did', 'vid'])
        cond1 = df_click['day'] >= cutoff_day
        cond2 = df_click['day'] <= 27
        recent_click = df_click[cond1 & cond2]
    else:  # 用于测试集
        print(f"加载曝光数据: {test_show_path}")
        test_show = load_csv_chunks(test_show_path, day=0)
        test_show_dedup = test_show.drop_duplicates(subset=['did', 'vid'])
    
    # 构建曝光映射
    user_exposure_dict = defaultdict(set)
    for did, vid in test_show_dedup[['did', 'vid']].values:
        user_exposure_dict[did].add(vid)
    
    user_exposure_counts = test_show_dedup.groupby(['did', 'vid']).size().to_dict()
    
    # 构建点击映射和特征
    user_click_dict = defaultdict(dict)
    for idx, row in recent_click.iterrows():
        user_click_dict[row['did']][row['vid']] = row['day']
    
    features = build_features(df_click, df_show, None, danmu_features, start_date=1, end_date=31)
    vid_ctr = features['vid_ctr']
    item_cid_popular = df_click.groupby('item_cid')['vid'].apply(lambda x: x.value_counts().index.tolist()).to_dict()
    global_hot_rank = {vid: rank for rank, vid in enumerate(df_click['vid'].value_counts().index)}
    
    # 构建用户历史播放记录
    user_history_vids = defaultdict(set)
    for did, vids in user_click_dict.items():
        user_history_vids[did].update(vids.keys())
    
    # 构建并训练播放率预测模型
    print("\n===== 用户的播放率预测 =====")
    # 构建模型特征
    df_model, feature_cols, encoders = build_play_rate_features(df_click, df_show, df_vid_info, df_did_features, None, danmu_features)
    
    # 训练模型
    model = train_play_rate_model(df_model, feature_cols)
    
    # 准备测试集特征
    print("准备测试集特征...")
    test_df = test_show_dedup.copy()
    test_df = test_df.merge(df_did_features, left_on='did', right_index=True, how='left')
    test_df = test_df.merge(df_vid_info, on='vid', how='left')
    test_df.fillna(0, inplace=True)
    
    # 预测播放率分数
    play_rate_dict = predict_play_rate(model, test_df, feature_cols, encoders)
    
    # 设置权重
    weights = weights or {
        'history_click': 0.1,
        'ctr': 0.4,
        'item_cid_popular': 0.1,
        'global_hot': 0.2,
        'play_rate_model': 0.4
    }
    
    # 并行处理批次
    batch_size = 10000
    batches = [test_dids[i:i+batch_size] for i in range(0, len(test_dids), batch_size)]
    
    # 参数
    args = (user_exposure_dict, user_exposure_counts, user_click_dict, vid_ctr, item_cid_popular, global_hot_rank, weights, df_vid_info)
    
    print(f"使用{PARALLEL_JOBS}个进程并行处理{len(batches)}个批次...")
    results = Parallel(n_jobs=PARALLEL_JOBS, verbose=10)(
        delayed(process_batch)((*args, batch, play_rate_dict, user_history_vids, danmu_features)) for batch in batches
    )
    
    # 合并结果并验证
    recall_results = {}
    for res in results:
        recall_results.update(res)
    

    for did in recall_results:
        if recall_results[did] is None:
            continue
            
        # 检查召回结果是否在曝光候选集中
        history_vids = user_history_vids.get(did, set())
        valid_candidates = set(user_exposure_dict.get(did, [])) - history_vids
        
        if recall_results[did] not in valid_candidates:
            # 选择曝光次数最多的视频
            if did in user_exposure_counts:
                valid_vids = [
                    vid for vid, count in user_exposure_counts[did].items()
                    if vid in valid_candidates
                ]
                if valid_vids:
                    recall_results[did] = max(
                        valid_vids, 
                        key=lambda x: user_exposure_counts[did][x]
                    )
                    continue
            
    
    print(f"召回完成，耗时: {time.time() - start_time:.2f}秒")
    return recall_results, user_exposure_dict, user_exposure_counts

def calculate_hit_rate(recall_results, true_play_pairs):
    hits = 0
    total = len(recall_results)
    
    for did, vid in recall_results.items():
        if (did, vid) in true_play_pairs:
            hits += 1
    
    hit_rate = hits / total if total > 0 else 0
    print(f"召回命中率: {hit_rate:.4f}")
    return hit_rate

if __name__ == "__main__":
    print("\n===== 开始召回 =====")
    start_time = time.time()
    print(f"开始时间: {time.ctime()}")
    
    # 加载数据
    df_click, df_show, df_vid_info, df_danmu, danmu_features = load_full_data()
    
    # 加载用户特征
    df_did_features, _ = load_did_features()

    # 划分训练集(1-28)和验证集(29-30)
    print("\n===== 划分训练集和验证集 =====")
    train_click = df_click[df_click['day'] <= 28].copy()
    val_click = df_click[df_click['day'].isin([29, 30])].copy()
    val_show = df_show[df_show['day'].isin([29, 30])].copy()
    
    # 在训练集上构建特征
    print("\n===== 在训练集上构建特征 =====")
    train_features = build_features(train_click, df_show, df_danmu, danmu_features, start_date=1, end_date=29)
    
    # 准备验证集真实播放数据
    true_play_pairs = set(zip(val_click[val_click['play_time'] > 0]['did'], 
                             val_click[val_click['play_time'] > 0]['vid']))
    print(f"验证集真实播放记录数: {len(true_play_pairs)}")
    
    # 评估既有曝光又有播放的用户
    eval_dids = list(set(val_show['did']) & {did for did, _ in true_play_pairs})
    print(f"待评估用户数: {len(eval_dids)}")
    
    # 6. 在训练集上训练模型并在验证集上评估
    print("\n===== 训练模型并在验证集上评估 =====")
    val_recall, _, _ = multi_strategy_recall(
        df_click=train_click,  # 使用1-28号数据
        df_show=df_show,
        df_vid_info=df_vid_info,
        df_did_features=df_did_features,
        test_dids=eval_dids,
        test_show_path=None,
        weights={
            'history_click': 0.2,
            'ctr': 0.4,
            'item_cid_popular': 0.1,
            'global_hot': 0.2,
            'play_rate_model': 0.4
        },
        custom_show_df=val_show,
        danmu_features=danmu_features
    )
    
    # 7. 计算验证集命中率
    hit_rate = calculate_hit_rate(val_recall, true_play_pairs)
    
    # 加载测试用户
    test_dids = load_csv_chunks(TEST_PRED_PATH, day=0)['did'].unique()  # 测试集设置day=0
    print(f"测试用户数量: {len(test_dids)}")
    
    # 定义加权（包含弹幕特征权重）
    weights = {
        'history_click': 0.1,    # 历史点击权重
        'ctr': 0.4,              # CTR权重
        'item_cid_popular': 0.1, # 同类热门权重
        'global_hot': 0.2,       # 全局热门权重
        'play_rate_model': 0.4   # 播放率模型权重
    }
    
    # 执行多策略召回
    test_recall, _, _ = multi_strategy_recall(
        df_click, df_show, df_vid_info, df_did_features, test_dids, TEST_SHOW_PATH, weights=weights, danmu_features=danmu_features
    )
    
    # 生成预测结果
    print("\n生成预测结果...")
    test_pred = pd.DataFrame({
        "did": test_dids,
        "vid": [test_recall.get(did, None) for did in test_dids]
    })

    test_pred['vid'] = test_pred['vid'].astype('Int64')
    
    # 保存结果
    output_path = "testA_pred_recall.csv"
    test_pred.to_csv(output_path, index=False)
    print(f"召回完成...")
    print(f"总耗时: {time.time() - start_time:.2f}秒")

