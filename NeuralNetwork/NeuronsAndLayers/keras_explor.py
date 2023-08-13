import numpy as np
import tensorflow as tf

print("Tensorflow version: ", tf.__version__)

def test_dense_linear():
    linear_layer = tf.keras.layers.Dense(units=1, activation="linear")
    weights = linear_layer.get_weights()
    # 初试状态无参数
    print("init weights: ", weights)
    #> init weights:  [] 
    
    # call() 的快速用法(__call__())
    # 使用仅有一个元素的数组来初始化此线性模型
    a1 = linear_layer(np.array([1.0]).reshape(1,1))
    print("a1: ", a1)
    #> a1:  tf.Tensor([[0.04257667]], shape=(1, 1), dtype=float32)

    w,b = linear_layer.get_weights() # 获取自动匹配的参数(weights, bias)
    print(f"w={w}, b={b}")
    #>w=[[-0.14207113]], b=[0.]
    print("Initialized weights: ", linear_layer.get_weights())
    #> Initialized weights:  [array([[0.04257667]], dtype=float32), array([0.], dtype=float32)]

    # set_weights takes a list of numpy arrays
    set_w = np.array([[200]])
    set_b = np.array([100])
    linear_layer.set_weights([set_w, set_b])

    # 推理
    X_train = np.array([[1.0], [2.0]], dtype=np.float32) 
    pred = linear_layer(X_train)
    print(f"tf.keras推理结果: {pred}")

def test_sequential_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(1, input_dim=1, activation="sigmoid", name="L1")
        ]
    )
    model.summary() # 打印模型层级概要

    logistic_layer = model.get_layer('L1')
    w,b = logistic_layer.get_weights()
    print(f"w={w}, b={b}")
    print(f"w.shape={w.shape}, b.shape={b.shape}")

    # 推理Senqutinel模型
    a1 = model.predict(np.array([1.0]).reshape(1,1))
    print(a1)

test_sequential_model()
