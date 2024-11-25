import pandas as pd
import numpy as np
import pickle
import csv

new_pkl_dir='..\data\samples_window_common.pkl'
with open(new_pkl_dir, 'rb') as f:
    original_data = pickle.load(f)
    print('original_data:', original_data.shape)
    # shape=original_data.shape
    # 将四维数据转换为二维数据
    # reshaped_data = original_data.squeeze()
#
# #
# df=pd.DataFrame(reshaped_data)
# # 将 DataFrame 保存到 CSV 文件
# csv_dir='E:\Papers\data_cnn_lstm\\CNN.csv'

# df.to_csv(csv_dir, index=False, header=True)
# print('pkl_shape:', shape)


# print('shape。')
