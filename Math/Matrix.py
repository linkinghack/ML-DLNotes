# encoding:utf-8
import numpy as np
from numpy.linalg import inv

# 创建矩阵
# 使用numpy创建矩阵有两种方式，一是使用matrix类，二是使用二维ndarray
# 主要区别在于默认的乘法不同，matrix类的默认乘法是矩阵乘法，ndarray默认乘法是Hadamard乘法（element-wise multiplication)
print("-- 创建矩阵 --")
A = np.matrix([[1,2,5],[3,4,6],[5,6,7]])
print("A: \n", A)

B = np.array(range(1, 7)).reshape(3,2)
print("B: \n",B)

# 矩阵乘法
print("A*A: \n",A * A) # matrix product
print("B*B \n",B * B) #element-wise

# ndarray 其它方法
zrs = np.zeros((3,2))
print("zeros: \n",zrs)

idt = np.identity(3, dtype=np.int64) # 创建单位矩阵
print(idt)
e = np.eye(3,4)
print("eye: \n", e)

# 矩阵中的向量提取
print("-- 矩阵中向量提取 --")
M = np.array(range(1,10)).reshape(3,3)
print("M: \n", M)
row1 = M[[0,2]] # 第一行和第三行，也可以使用 M[[True, False, True]]
print(row1)
col1 = M[:, [1,2]] #第二列和第三列， 或者使用M[:, [False, True, True]]
print(col1)


# 矩阵运算
print("-- 矩阵运算 --")
# 转置矩阵, 不管是Matrix类还是ndarray都可以用一以下两种方式
transposeOfMatrixObject = A.T
transpostOf2dArray = np.transpose(B)
print("Transpose: ")
print(transposeOfMatrixObject)
print(transpostOf2dArray)
# 矩阵加减法，都是对应元素想加减
print(A - M) # 矩阵减法
print(A + M) # 矩阵加法

# 矩阵乘法
print('-- 矩阵乘法 --')
print(A.dot(M))
print(M * M) # element-wise
print(M.dot(M)) # matrix produce

# 逆矩阵
# 使用numpy.linalg库提供的inv方法
print("-- 矩阵求逆 --")
print("A-inverse: \n",inv(A))
print("A * inv(A): \n", A*inv(A))