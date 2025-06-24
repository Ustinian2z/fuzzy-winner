import os
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import r2_score
import numpy as np

# 播放时间路径
root_folder = './用户历史播放数据'

# 用于存储所有播放数据列表
user_video_playtime_data = []

for i in range(1, 31):
    day_str = f"day{str(i).zfill(2)}"
    folder_path = os.path.join(root_folder, day_str)
    file_path = os.path.join(folder_path, f"{day_str}_data.csv")

    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path)
            data['date_day'] = i
            user_video_playtime_data.append(data)
            # print(f"成功读取 {file_path} 的数据")
        except Exception as e:
            print(f"读取 {file_path} 时出现错误: {e}")
    else:
        print(f"{file_path} 文件不存在")

# 合并数据
if user_video_playtime_data:
    user_video_playtime_data_combined = pd.concat(user_video_playtime_data, ignore_index=True)
    print("所有数据已合并")
else:
    print("未读取到任何数据")


# 读取vid_info表
vid_info_folder = './vid_info/vid_info_table.csv'
vid_info = pd.read_csv(vid_info_folder)

# 读取did_features表
did_features_folder = './did_features/did_features_table.csv'
did_features = pd.read_csv(did_features_folder)

# 特征聚合
add_data = pd.merge(user_video_playtime_data_combined, vid_info, on=['vid','item_cid'], how='left')
add_data['label_mae'] = add_data['play_time'] / add_data['item_duration']
add_data = pd.merge(add_data, did_features, on=['did'], how='left')

add_data = pl.from_pandas(add_data)


# 训练、验证、测试集划分
train_df = add_data.filter(pl.col("date_day")<31)
valid_df = add_data.filter((pl.col("date_day")==30))
test_df = add_data.filter(pl.col("date_day")==30)
print(train_df.shape,test_df.shape,valid_df.shape)


features = [
    'vid', 'item_cid', 'item_type', 'item_duration', 
    'item_assetSource', 'item_classify', 'item_isIntact', 'item_serialno',
    'sid', 'stype'
]

features += [f'f{i}' for i in range(87)]
label=['play_time']


# 提取特征和标签
X_train = train_df[features].to_numpy()
y_train = train_df[label].to_numpy()

# 获取训练集和验证集的用户ID集合
train_did_set = set(train_df['did'].unique())

# 创建 LightGBM 
train_data = lgb.Dataset(X_train, label=y_train)
# 设置 LightGBM 参数
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 600,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed':42,
}

num_boost_round = 2000

# 模型训练
print("开始模型训练...")
model_valid = lgb.train(params,
                       train_data,
                       num_boost_round=num_boost_round)

print("模型训练完成...")

# 时长预测
testA_folder = './testA_pred_recall.csv'
testA = pd.read_csv(testA_folder)

# 特征构建
print("开始对测试集时长预测...")
A_data = pd.merge(testA, vid_info, on=['vid'], how='left')
A_data = pd.merge(A_data, did_features, on=['did'], how='left')
A_data = A_data.drop(['did','f87'], axis=1)

testA_pred = model_valid.predict(A_data)
testA_pred = testA_pred/A_data['item_duration']


# 最终文件输出
if isinstance(testA_pred, pd.Series):
    testA_pred = testA_pred.to_frame(name='completion_rate')

did_vid = testA[['did', 'vid']]
merged_df = pd.concat([did_vid, testA_pred], axis=1)

# 保存结果
merged_df.to_csv('result.csv', index=False)

print("结果生成完成...")


=========== 分割线 ===========

####### 结果评估评估参考
resultA_folder = './testA_result.csv'
resultA = pd.read_csv(resultA_folder)
resultA_dict = resultA.set_index(['did', 'vid'])['completion_rate'].to_dict()

merged_df['key'] = list(zip(merged_df['did'], merged_df['vid']))
merged_df['exists'] = merged_df['key'].isin(resultA_dict)

# 计算分数
matched = merged_df[merged_df['exists']].copy()
matched['resultA_rate'] = matched['key'].map(resultA_dict)
matched['score'] = 1 / (1 + np.sqrt(abs(matched['resultA_rate'] - matched['completion_rate']) / matched['resultA_rate']))

total_score = matched['score'].sum()/len(resultA)

print(f"最终分数: {total_score}")


=========== 分割线 ===========

##### 爆点视频特征提取参考
import cv2
import numpy as np

def extract_orb_features(video_path):
    # 初始化ORB特征检测器
    orb = cv2.ORB_create()

    # 视频文件
    cap = cv2.VideoCapture(video_path)

    all_descriptors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测关键点
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is not None:
            all_descriptors.extend(descriptors)

    # 释放视频文件
    cap.release()

    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        video_feature = np.mean(all_descriptors, axis=0)
        return video_feature
    else:
        return None


# 视频文件路径
video_path = './932540.mp4'

# 提取视频特征
video_feature = extract_orb_features(video_path)







