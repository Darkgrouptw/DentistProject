import tensorflow as tf
import numpy as np

# 先測試 Tensorflow 的變數
TensorA = tf.placeholder(tf.int32, 1, "TensorA")
TensorB = tf.placeholder(tf.int32, 1, "TensorB")
TensorC = tf.add(TensorA, TensorB)
Sess = tf.Session()
print("Test2")

# 測試
def TestPrint():
    print("TestPY")
def AddTest(a, b):
    return a + b
def TensorTest(a, b):
    # global TensorA, TensorB, TensorC, Sess
    feed_dict = {
        TensorA: [a],
        TensorB: [b]
    }
    result = Sess.run(TensorC, feed_dict=feed_dict)
    print("Tensor Add: ", result)

def NumpyArrayTest(arrayData):
    print(arrayData.shape)
    print(arrayData)

def NumpyOperationTest(arrayData):
    print("OperationTest: ", arrayData.shape)
    return arrayData * 3

def NumpyArrayConcatenate(arrayData1, arrayData2):
    print("Array 1: ", arrayData1)
    print("Array 2: ", arrayData2)
    return np.concatenate([arrayData1, arrayData2], axis=0).reshape([1, -1])