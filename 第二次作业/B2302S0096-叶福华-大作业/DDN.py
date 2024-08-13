
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Dataset import train_dataset, test_dataset


trian_batch_size = 200
test_batch_size= 1000
hidden_size = 125
learning_rate = 0.00001
epoch = 200
# 加载神经网络数据
train_dataloader = DataLoader(dataset=train_dataset, batch_size=trian_batch_size, shuffle=True, num_workers=0,
                              drop_last=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=False)


# 搭建神经网络
class Curvature(nn.Module):
    def __init__(self):
        super(Curvature, self).__init__()
        self.model = nn.Sequential(

            nn.Linear(51, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 50),
            nn.ReLU(),

        )


    def forward(self, z):
        z = z.to(torch.float32)
        z = self.model(z)
        # out = Fun.softmax(z, dim=1)
        return z


curvature = Curvature()

# curvature = curvature.cuda() GPU
# #损失函数
loss_fn = nn.L1Loss()
# # loss_fn = loss_fn.cuda() GPU
# # 优化器
learning = learning_rate
# optimizer = torch.optim.RMSprop(curvature.parameters(), lr=learning)  # 该优化器优于Adam 固支
optimizer = torch.optim.Adam(curvature.parameters(), lr=learning_rate)  # 使用Adam优化器，并设置学习率 一端固支一端简支优化器
# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试次数
total_test_step = 0

for i in range(epoch):
    print("-------第{} 轮训练开始--------".format(i + 1))

    # 训练步骤开始
    for data in train_dataloader:
        x, y = data
        x = x.view([-1, 51])
        y = y.to(torch.float32)


        outputs = curvature(x)
        loss = loss_fn(outputs, y)  # loss 是一个样本的loss 还是整个batch-size的loss？？？

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 500 == 0:
        # 当训练次数遇到百时才打印
            print("训练次数：{}， Loss: {}".format(total_train_step, loss))
        #测试步骤开始
        total_test_loss = 0
        with torch.no_grad():
            for data1 in test_dataloader:
                x, y = data1
                x = x.view([-1, 51])
                y = y.to(torch.float32)


                output1 = curvature(x)
                # outputs = outputs.clamp(5e-4, 0.01)
                loss = loss_fn(output1, y)
                total_test_loss = total_test_loss + loss.item()
                print("整体测试集上的Loss: {}".format(total_test_loss))