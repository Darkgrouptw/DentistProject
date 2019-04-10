# @package 2DNetwork
# 用來 2D Convolution 的捲機神經網路

import sys
if sys.platform != "win32":
    import setGPU
import tensorflow as tf
from tensorflow.layers import conv2d, batch_normalization, max_pooling2d, dropout, flatten, dense
from tensorflow.nn import relu
from tqdm import tqdm
import numpy as np

tf.set_random_seed(1)

class Network_Prob:
    def __init__(self, sizeX, sizeY, OutClass, lr = 1e-3, kernelSize = 3, logdir = "./logs", IsDebugGraph = False):
        # 神經網路大小
        self.SizeX = sizeX
        self.SizeY = sizeY
        self.OutClass = OutClass

        # 初始化網路
        self.InitNetwork(lr, kernelSize)
        self.Session = tf.Session()
        self.Session.run(tf.global_variables_initializer())

        # 寫檔
        self.LogWriter = tf.summary.FileWriter(logdir)
        if IsDebugGraph:
            self.LogWriter.add_graph(self.Session.graph)
            print("Debug Output")

    # 初始化網路
    def InitNetwork(self, lr, kernelSize):
        # 常數設定
        layer1_Units = 16
        layer2_Units = 32
        layer3_Units = 32
        layer1_KernelSize = layer2_KernelSize = layer3_KernelSize = kernelSize
        layer1_PaddingCount = layer2_PaddingCount = layer3_PaddingCount = 1
        layer1_MaxpoolCount = layer2_MaxpoolCount = layer3_MaxpoolCount = 2

        # 輸入
        self.InputData = tf.placeholder(tf.float32, [None, self.SizeY, self.SizeX, 1], name="InputLayer")
        self.LabeledClass = tf.placeholder(tf.float32, [None, self.OutClass], name="LabeledData")

        # Conv2D
        layer1 = self._AddConvoluationLayer(self.InputData, layer1_Units, layer1_KernelSize, layer1_PaddingCount, layer1_MaxpoolCount, "Layer1")
        layer2 = self._AddConvoluationLayer(layer1, layer2_Units, layer2_KernelSize, layer2_PaddingCount, layer2_MaxpoolCount, "Layer2")
        layer3 = self._AddConvoluationLayer(layer2, layer3_Units, layer3_KernelSize, layer3_PaddingCount, layer3_MaxpoolCount, "Layer3")

        # 攤平
        layerFlatten = flatten(layer3, "Layer_Flatten")

        # Dense
        layer1Dense = self._AddDenseLayer(layerFlatten, 1024, "Layer1Dense")
        layer2Dense = self._AddDenseLayer(layer1Dense, 256, "Layer2Dense")
        layer3Dense = self._AddDenseLayer(layer2Dense, 64, "Layer3Dense")

        # 預測
        self.PredictProb = dense(layer3Dense, self.OutClass, name="PredictProb")
        # self.PredictClassProb = tf.nn.softmax(predict, name= "PredictProb")
        # self.PredictClass = tf.argmax(predictClassProb, name="PredictImg")
        print(self.PredictProb)

        with tf.name_scope("Loss"):
            # loss = tf.losses.softmax_cross_entropy(self.LabeledClass, predict)
            loss = tf.losses.mean_squared_error(self.LabeledClass, self.PredictProb)
            self.Optimzer = tf.train.AdamOptimizer(lr).minimize(loss)

        # Log
        self.LossTensor = tf.summary.scalar("LossTensor", loss)
        # self.ExampleTensor = tf.summary.image("Example", self.PredictImg)

    # 預測
    def Train(self, DM, epochNum, batchSize):
        for i in tqdm(range(epochNum + 1)):
            # 抓取資料
            Train_BatchData, Labeled_BatchData = DM.BatchTrainData(batchSize)
            feed_dict = {
                self.InputData: Train_BatchData,
                self.LabeledClass: Labeled_BatchData
            }

            # 紀錄 Train 的結果
            if i % 100 == 0:
                feed_dict_FirstN = {
                    self.InputData: DM.TestFirstNBoxOfTrainData(3),
                }
                lossSummary = self.Session.run(self.LossTensor, feed_dict=feed_dict)
                # exampleSummary = self.Session.run(self.ExampleTensor, feed_dict=feed_dict_FirstN)

                # mergeSummary = tf.Summary.merge([lossSummary, exampleSummary])
                self.LogWriter.add_summary(lossSummary, i)
                # self.LogWriter.add_summary(exampleSummary, i)

            # 更新
            self.Session.run(self.Optimzer, feed_dict=feed_dict)

    # 預測資料
    def Predict(self, Data):
        # Data
        # print(Data.shape[])
        topTime = int(np.ceil(Data.shape[0] / 128))
        for i in tqdm(range(topTime)):
            feed_dict = {
                self.InputData: Data[i * 128: (i + 1) * 128]
            }
            resultTemp = self.Session.run(self.PredictProb, feed_dict=feed_dict)

            if i == 0:
                result = resultTemp
            else:
                result = np.concatenate([result, resultTemp], axis=0)
        # print(
        return result

    # 圖片放進去做 Debug 用
    # def PredictImg

    # 存 Weight
    def SaveWeight(self,  logdir="./logs"):
        saver = tf.train.Saver()
        saver.save(self.Session, logdir + "/Model.ckpt")

    # Load Weight
    def LoadWeight(self, logdir="./logs"):
        saver = tf.train.Saver()
        saver.restore(self.Session, logdir + "/Model.ckpt")

    # 清除記憶體
    def Release(self):
        # 清除之前的 Graph
        self.Session.close()
        tf.reset_default_graph()

    # Helper Function
    def _AddConvoluationLayer(self, inpuTensor, units, kernel_size, padding, maxpoolSize, name):
        with tf.name_scope(name):
            layer_conv = conv2d(inpuTensor, units, kernel_size, padding, "same", name= name + "_Conv")
            layer_bn = batch_normalization(layer_conv, name= name + "_BN")
            layer_act = relu(layer_bn, name= name + "_Relu")
            layer_maxpool = max_pooling2d(layer_act, maxpoolSize, maxpoolSize, "same", name= name + "_Maxpool")
            layer_dropout = dropout(layer_maxpool, 0.5, name= name + "_DropOut")
        return layer_dropout

    def _AddDenseLayer(self, inputTensor, units, name):
        with tf.name_scope(name):
            layer_dense = dense(inputTensor, units, name= name + "_Dense")
            layer_bn = batch_normalization(layer_dense, name= name + "_BN")
            layer_act = relu(layer_bn, name= name + "Relu")
            layer_dropout = dropout(layer_act, name= name + "_DropOut")
        return layer_dropout