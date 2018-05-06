# 两层全链接网络
# pytorch 官方示例
import numpy as np

# N为样本大小； D_in为样本维度
# H为隐藏层维度； D_out 为输出维度(分类数)
N,D_in, H, D_out = 64,1000,100,10

#生成随机样本
x = np.random.randn(N,D_in)
y = np.random.randn(N,D_out)

#生成随机权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    #前向传播：计算Y的预测值
    h = x.dot(w1)
    h_relu = np.maximum(h,0) #ReLU 激活函数
    y_pred = h_relu.dot(w2)

    #计算误差并输出
    loss = np.square(y_pred - y).sum()
    print(t,loss)

    #更新权重；
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2