import torch
import numpy as np
import pandas as pd

class Dataset():

    def __init__(self, flag, csv_file1, csv_file2):
        self.csv_file1 = csv_file1
        self.csv_file2 = csv_file2
        self.flag = flag
        self.X_data = []
        self.Y_data = []
        self.load_data()

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        input_data = self.X_data[idx]
        output_data = self.Y_data[idx]
        return input_data, output_data

    def load_data(self):

        df_train1 = pd.read_csv(self.csv_file1,
                                skiprows=lambda x: x not in range(0, 5001))  # 第0行指的表头的下一行  第0~999是数据
        df_train2 = pd.read_csv(self.csv_file2,
                                skiprows=lambda x: x not in range(0, 5001))
        df_train1 = np.asarray(df_train1)
        df_train2 = np.asarray(df_train2)

        x = df_train1[0:5000, 0:51]  # 0：1000指的是 0~999行
        y = df_train2[0:5000, 0:50]
        x = torch.tensor(x)
        y = torch.tensor(y)

        if self.flag == 'train':

            self.X_data = x[0:4000, 0:51]
            self.Y_data = y[0:4000, 0:50]

        if self.flag == 'test':
            self.X_data = x[4000:5000, 0:51]
            self.Y_data = y[4000:5000, 0:50]


train_dataset = Dataset('train', r'E:\Microneedle\pythonProject1\0.5appInput_Files\1t_all.csv',
                        r'E:\Microneedle\pythonProject1\0.5appInput_Files\1c_all.csv')

test_dataset = Dataset('test', r'E:\Microneedle\pythonProject1\0.5appInput_Files\1t_all.csv',
                       r'E:\Microneedle\pythonProject1\0.5appInput_Files\1c_all.csv')