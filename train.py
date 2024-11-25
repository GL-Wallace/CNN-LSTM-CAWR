import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import math
import models
import data_helper
import config as cfg
import utils
import learning_rate

def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup: True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step >0 and epochs>0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x)/(warmup_epochs*num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1-(x - warmup_epochs * num_step)/((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def get_data_loader(x_data, y_data, train_idx, test_idx):
    train_dataset = data_helper.Dataset(x_data=x_data, y_data=y_data, data_index=train_idx, transform=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = data_helper.Dataset(x_data=x_data, y_data=y_data, data_index=test_idx, transform=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def get_data_loader_cnnlstm(x_data_cnn, x_data_lstm, y_data, train_idx, test_idx):
    train_dataset = data_helper.DatasetCNNLSTM(x_data_cnn=x_data_cnn, x_data_lstm=x_data_lstm, y_data=y_data, data_index=train_idx, transform=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = data_helper.DatasetCNNLSTM(x_data_cnn=x_data_cnn, x_data_lstm=x_data_lstm, y_data=y_data, data_index=test_idx, transform=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader

# get_model_and_dataloader_without_evi altered by Wallace Zhang
# Date: 12/6/2023 14:30
def get_model_and_dataloader(x_cnn_common, x_ts_lsp,  y, train_idx, test_idx, x_cnn_t_c, x_cnn_bands_t_c):
    if cfg.model_name == 'CNN':
        model = models.ConvNet(num_channels=cfg.num_channels)
        train_loader, test_loader = get_data_loader(x_data=x_cnn_common, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'CNN-Terrain-Climate':
        model = models.ConvTerrianClimate(num_channels=cfg.num_channels)
        train_loader, test_loader = get_data_loader(x_data=x_cnn_t_c, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'CNN_s2_t_c':
        model = models.ConvTerrianClimate(num_channels=cfg.num_channels)
        train_loader, test_loader = get_data_loader(x_data=x_cnn_bands_t_c, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'LSTM_lsp':
        model = models.SimpleLSTM(input_size=cfg.lstm_input_size_evi+cfg.lstm_input_size_lsp, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers, dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader(x_data=x_ts_lsp, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'CNN_t_c-LSTM_lsp':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_lsp, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader_cnnlstm(x_data_cnn=x_cnn_t_c, x_data_lstm=x_ts_lsp, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'CNN_s2-LSTM_lsp':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_lsp, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader_cnnlstm(x_data_cnn=x_cnn_common, x_data_lstm=x_ts_lsp, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'CNN_s2_t_c-LSTM_lsp':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_lsp, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader_cnnlstm(x_data_cnn=x_cnn_bands_t_c, x_data_lstm=x_ts_lsp, y_data=y, train_idx=train_idx, test_idx=test_idx)
    else:
        print('Model name is not valid.')
        sys.exit(0)
    return model, train_loader, test_loader


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


train_RMSE = []
train_R2 = []
test_RMSE = []
test_R2 = []
train_loss_values = []
test_loss_values = []

def train_model(model, opt,  train_loader, test_loader):
    torch.cuda.empty_cache()
    torch.manual_seed(cfg.rand_seed)
    torch.cuda.manual_seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)

    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lf = lambda x, y= cfg.epochs: (((1 + math.cos(x * math.pi / y)) / 2) ** 1.0) * 0.65 + 0.35
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    scheduler = learning_rate.CosineAnnealingWarmbootingLR(optimizer, epochs=cfg.epochs, steps=opt.cawb_steps, step_scale=0.8,
                                         lf=lf, batchs=len(train_loader), warmup_epoch=0, epoch_scale=4.0)
    learning_rate.plot_lr_scheduler(optimizer, scheduler, epochs=cfg.epochs, save_dir='../lr.png')
    criterion = nn.MSELoss()
    lrs = []
    best_rmse, best_mae, best_r2 = np.inf, np.inf, -np.inf
    best_epoch = 1

    for epoch in range(1, cfg.epochs + 1):
        # print('epoch: {}'.format(epoch))
        model.train()
        train_loss_list = []
        for batch_idx, data_input in enumerate(train_loader):
            if epoch == 1 and batch_idx == 0:
                print('input_data_shape:')
                for data in data_input:
                    print('data_shape:', data.shape)
                print()
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            # global_step = batch_idx + (epoch - 1) * int(len(train_loader.dataset) / len(inputs)) + 1
            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                y_pred = model(x_input)
            # print('y_input.shape: ', y_input.shape, 'y_pred.shape: ', y_pred.shape)
            y_input = y_input.double()
            y_pred = y_pred.double().unsqueeze(dim=1)
            loss = criterion(y_input, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            loss_val = loss.cpu().data.numpy()
            train_loss_list.append(loss_val)
        train_loss_mean = np.mean(train_loss_list)
        train_loss_values.append(train_loss_mean)

        print('train_loss = {:.3f}'.format(train_loss_mean))
        if epoch % cfg.eval_interval != 0:
            continue
        print('epoch: {}'.format(epoch))
        model.eval()
        y_input_list = []
        y_pred_list = []
        for batch_idx, data_input in enumerate(train_loader):
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                y_pred = model(x_input)

            y_pred_list.extend(y_pred.data.cpu().numpy())
            y_input_list.extend(y_input.data.cpu().numpy())
        rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))
        mae = metrics.mean_absolute_error(y_input_list, y_pred_list)
        r2 = metrics.r2_score(y_input_list, y_pred_list)
        print('Train_RMSE = {:.3f}  Train_MAE = {:.3f}  Train_R2 = {:.3f}'.format(rmse, mae, r2))
        train_RMSE.append(rmse)
        train_R2.append(r2)

        y_input_list = []
        y_pred_list = []
        test_loss_list = []
        for batch_idx, data_input in enumerate(test_loader):
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                y_pred = model(x_input)
            y_pred = y_pred.double().unsqueeze(dim=1)
            loss = criterion(y_input, y_pred)
            loss_val = loss.cpu().data.numpy()
            test_loss_list.append(loss_val)

            y_pred_list.extend(y_pred.data.cpu().numpy())
            y_input_list.extend(y_input.data.cpu().numpy())
        rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))
        mae = metrics.mean_absolute_error(y_input_list, y_pred_list)
        r2 = metrics.r2_score(y_input_list, y_pred_list)
        test_loss_mean = np.mean(test_loss_list)
        test_loss_values.append(test_loss_mean)

        print('test_loss = {:.3f}'.format(test_loss_mean))

        torch.save(model.state_dict(), cfg.model_save_pth)
        print('Test_RMSE  = {:.3f}  Test_MAE  = {:.3f}  Test_R2  = {:.3f}'.format(rmse, mae, r2))
        test_RMSE.append(rmse)
        test_R2.append(r2)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--scheduler_lr', type=str, default='cawb', help='the learning rate scheduler, cos/cawb')
    parser.add_argument('--cawb_steps', nargs='+', type=int, default=[200,  400,  600,  800, 1000], help='the cawb learning rate scheduler steps')
    opt = parser.parse_args()

    # Basic setting
    device = torch.device('cuda:0' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # Load data
    x_cnn_common, x_cnn_t_c, x_ts_lsp, x_cnn_bands_t_c, y = utils.generate_xy()
    # print('x_cnn_common.shape: {}; x_cnn_t_c.shape: {};  x_ts_lsp.shape: {}; x_cnn_bands_t_c.shape: {}; y_length: {}\n'.format(x_cnn_common.shape, x_cnn_t_c.shape, x_ts_lsp.shape, x_cnn_bands_t_c.shape, len(y)))
    # sys.exit(0)

    # Build the model
    train_idx = utils.load_pickle(cfg.f_train_index)
    test_idx = utils.load_pickle(cfg.f_test_index)
    model, train_loader, test_loader = get_model_and_dataloader(x_cnn_common = x_cnn_common, x_cnn_t_c=x_cnn_t_c, x_ts_lsp=x_ts_lsp, x_cnn_bands_t_c= x_cnn_bands_t_c, y=y, train_idx=train_idx, test_idx=test_idx)
    if cfg.device == 'cuda':
        model = model.cuda()
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, model))

    # Train the model
    input('Press enter to start training...\n')
    print('START TRAINING\n')
    train_model(model, opt, train_loader, test_loader)

    plt.plot(train_RMSE, label='train_RMSE')
    plt.plot(train_R2, label='train_R2')
    plt.plot(test_RMSE, label='test_RMSE')
    plt.plot(test_R2, label='test_R2')
    plt.legend()
    plt.show()

    # 创建一个 x 轴的数值，比如可以使用训练步骤的索引作为 x 轴
    x = range(1, len(train_loss_values) + 1)
    x1 = range(1, len(test_loss_values) + 1)

    # 使用 Matplotlib 绘制损失值变化曲线
    plt.plot(x, train_loss_values, label='Train Loss')
    plt.plot(x1, test_loss_values, label='Test Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
