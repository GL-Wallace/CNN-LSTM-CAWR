import numpy as np
import pickle
from sklearn.model_selection import KFold
import csv
import pandas as pd
import os

csv_direct='../data/sample_values.csv'
df = pd.read_csv(csv_direct)
data_array = df.values
data_indices = np.arange(len(data_array))

# 设置k-fold交叉验证的参数
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

data_dir = "../data/train_test_idx"
# 分割数据集并保存索引到pickle文件
for fold, (train_indices, test_indices) in enumerate(kf.split(data_indices)):
    train_file = os.path.join(data_dir, f"train_{fold+1}.pkl")
    test_file = os.path.join(data_dir,f"test_{fold+1}.pkl")

    train_data = data_indices[train_indices]
    test_data = data_indices[test_indices]
    train_reshaped_data = np.expand_dims(train_data, axis=-1)
    test_reshaped_data = np.expand_dims(test_data, axis=-1)
    print('Train data length:',len(train_data), '; Test data length:', len(test_data))
    with open(train_file, 'wb') as f:
        pickle.dump(train_reshaped_data, f)

    with open(test_file, 'wb') as f:
        pickle.dump(test_reshaped_data, f)

print("Finished all these files!")