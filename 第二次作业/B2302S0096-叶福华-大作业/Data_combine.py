import torch
import pandas as pd
import os


def data_combine(flag,Folder_Path,SaveFile_Name):

    # x坐标数据所在文件夹位置
    global var1, var2
    SaveFile_Path = Folder_Path
    SaveFile_Name = SaveFile_Name  # 合并后要保存的文件名

    if flag == 'x':  # x坐标
        var1 = 'Beam_No'
        var2 = 'Displacement_x'
    if flag == 'w':  # w坐标
        var1 = 'Beam_No'
        var2 = 'Displacement_w'
    if flag == 'c':  # 曲率值
        var1 = 'Beam_No'
        var2 = 'Displacement_Curvature'
    if flag == 't':  # 厚度分布
        var1 = 'RandData_No'
        var2 = ''

    os.chdir(Folder_Path)  # 修改当前工作目录
    # file_list = os.listdir()  # 将该文件夹下的所有文件名存入一个列表

    for i in range(1, 5001):
        df = pd.read_csv(Folder_Path + '/' + var1 + str(i) + var2 + '.csv')
        df = df.T  # 将该文件夹下所有文件的一列数据进行转置变成行数据
        df.to_csv(Folder_Path + '/' + '1' + var1 + str(i) + var2 + '.csv')

    df = pd.read_csv(Folder_Path + '/' + '1' + var1 + str(1) + var2 + '.csv')  # 读取第一个文件夹并包含表头

    # 将读取的第一个csv文件写入合并后的文件保存，index=False的意思是不读取序列
    df.to_csv(SaveFile_Path + '/' + SaveFile_Name, encoding="utf_8_sig", index=False)

    # 合并
    for i in range(2, 5001):
        print(i)
        data = pd.read_csv(Folder_Path + '/' + '1' + var1 + str(i) + var2 + '.csv')
        df = df.append(data)
    df.to_csv(SaveFile_Path + '/' + SaveFile_Name, index=False)


flag = 't'
Folder_Path = r"E:\Microneedle\pythonProject1\0.5appInput_Files"
SaveFile_Name = r'1t_all.csv'
data_combine(flag, Folder_Path, SaveFile_Name)

