import tensorflow as tf

# 先測試 Tensorflow 的變數
TensorA = tf.placeholder(tf.int32, 1, "TensorA")
TensorB = tf.placeholder(tf.int32, 1, "TensorB")
TensorC = tf.add(TensorA, TensorB)
Sess = tf.Session()

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

print("Test2")