# coding=utf-8
import os
import pickle

# model hyper-parameters
device = 'cuda'  # 'cpu' or 'cuda'
rand_seed = 188

# ['CNN', 'CNN-Terrain-Climate', 'CNN_s2_t_c', 'LSTM_lsp', 'CNN_t_c-LSTM_lsp', 'CNN_s2-LSTM_lsp', 'CNN_s2_t_c-LSTM_lsp' ]
model_name = 'CNN'  # altered by wallace

# hyper-parameter of CNNw
# num_channels = 10
num_channels = 1 # altered_by me

# hyper-parameter of LSTM (small values for parameters for initializing the model training)
lstm_input_size_evi = 0     # feature_size of EVI time series
# lstm_input_size_lsp = 11    # feature_size of LSP (phenology) time series
lstm_input_size_lsp = 9  # feature_size of LSP (phenology) time series
lstm_hidden_size = 6        # hidden size and layers do not need to be large for LSTM
lstm_num_layers = 2
lstm_dropout = 0

# hyper-parameter for training
lr = 0.1

batch_size = 32
epochs = 800    # need to consider early stopping to avoid overfitting
eval_interval = 10

data_dir = './data/'
log_dir = './log/'
f_df_samples = os.path.join(data_dir, 'sample_values.csv')   # user need to assign the filename of the sample data (including columns of the target soil property, e.g. soil organic carbon values)
target_var_name = 'SOM'     # the column name for the target property (y) that needs to be predicted
f_data_DL_common = os.path.join(data_dir, 'allinfo_CNN.pkl')
# f_data_DL_terrain_climate = os.path.join(data_dir, 'temp_terrain_climate_9.pkl')     # the pickle file of the input data (X) for CNN (i.e. climate and topographic data with spatially contextual information)
f_data_DL_terrain_climate = os.path.join(data_dir, 'temp_terrain_test.pkl')     # the pickle file of the input data (X) for CNN (i.e. climate and topographic data with spatially contextual information)
f_data_DL_lsp = os.path.join(data_dir, 'temp_lstm_test.pkl')               # the pickle file of the input data (X) for LSTM (i.e. phenological data with temporally dynamic information)

train_test_id = 5
f_train_index = os.path.join(data_dir, 'train_test_idx', 'train_{}.pkl'.format(train_test_id))  # the pickle file of the sample id list for the training set
f_test_index = os.path.join(data_dir, 'train_test_idx', 'test_{}.pkl'.format(train_test_id))    # the pickle file of the sample id list for the testing set

model_save_pth = './model/{}_{}.pth'.format(model_name, train_test_id)  # the save path of the model parameters

