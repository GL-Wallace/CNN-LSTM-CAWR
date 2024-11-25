import torch
import pickle
import pandas as pd
import numpy as np

# train_index_direct=\
#     'E:\Papers\A CNN-LSTM Model for Soil Organic Carbon Content Prediction with Long Time Series of MODIS-Based Phenological Variables\CNN-LSTM_for_DSM-main\data\samples_ts_lsp.pkl'
# # 步骤 1: 加载原始的pkl文件，获取数据
# with open(train_index_direct, 'rb') as f:
#     original_data = pickle.load(f)
#     shape=original_data.shape
#     print('Original_shape:', shape)

csv_direct="E:\Papers\paper1data\All.csv"
df = pd.read_csv(csv_direct)
data_array = df.values  # 将DataFrame转换为数组
# reshaped_data=np.expand_dims(data_array, axis = -1)
# print('数据类型：', reshaped_data.shape)
# # print('data_array：', data_array)
# # print('Current_数据shape：', data_array.shape)
reshaped_data = np.reshape(data_array, (188, 73, 1))
# new_data = torch.randn_like(original_data)


# 步骤 3: 将新的数据保存回pkl文件中
new_pkl_dir='..\data\\allinfo_CNN.pkl'
with open(new_pkl_dir, 'wb') as f:
    pickle.dump(reshaped_data, f)

print('数据已成功替换并保存回pkl文件中。')