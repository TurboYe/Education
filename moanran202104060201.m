%默安然_202104060201
clear,clc;
%向量形式给出已知点的坐标
%X = [1,2,3,4,5,6,7,8,9,10,11,12]; 
%Y = [5,8,9,15,25,29,31,30,22,25,27,29]; 

%自定义输入
X = input('请依次输入已知点的X坐标：');
Y = input('请依次输入已知点的Y坐标：');

%随机值输入
%X = randi([1,50],1,4);
%Y = randi([1,100],1,4);

n = length(X); %确定已知点数量n
L = ones(n,n); %初始化基函数系数矩阵L
for i = 1:n;
    A = 1;
    for j = 1:n
        if i ~= j
            B = [1,X(j)];
            A = conv(A,B)./(X(i)-X(j)); %逐项求基函数系数
        end
    end
    L(i,:) = A; %将基函数系数赋值给L
end
L1 = Y*L; %拉格朗日函数系数L1
disp('拉格朗日多项式系数为：')
disp(L1)

