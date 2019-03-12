# @package 3DNetwork
# 用來 3D Convolution 的捲機神經網路
import tensorflow as tf
from tensorflow.layers import conv3d, batch_normalization, max_pooling3d, dropout, conv2d_transpose, conv2d
from tensorflow.nn import relu
from tqdm import tqdm
import numpy as np

tf.set_random_seed(1)

class Network3D():
    def __init__(self, sizeX, sizeY, sizeZ, OutClass, lr = 1e-3, kernelSize = 3, logdir = "./logs", IsDebugGraph = False):
        # 神經網路大小
        self.SizeX = sizeX
        self.SizeY = sizeY
        self.SizeZ = sizeZ
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
        self.InputData = tf.placeholder(tf.float32, [None, self.SizeZ, self.SizeY, self.SizeX, 1], name="InputLayer")
        self.LabeledData = tf.placeholder(tf.float32, [None, self.SizeY, self.SizeX, self.OutClass], name="LabeledData")

        # Conv
        layer1 = self._AddConvoluationLayer(self.InputData, layer1_Units, layer1_KernelSize, layer1_PaddingCount, layer1_MaxpoolCount, "Layer1")
        layer2 = self._AddConvoluationLayer(layer1, layer2_Units, layer2_KernelSize, layer2_PaddingCount, layer2_MaxpoolCount, "Layer2")
        layer3 = self._AddConvoluationLayer(layer2, layer3_Units, layer3_KernelSize, layer3_PaddingCount, layer3_MaxpoolCount, "Layer3")

        # 中間
        layer_mid = self._MidTo2D(layer3, int(self.SizeZ / 2 / 2 / 2))

        # DeConv
        layer1_Upsample = self._AddUpSampleLayer(layer_mid, layer3_Units, layer3_KernelSize, layer3_MaxpoolCount, "Layer1Up")
        layer2_Upsample = self._AddUpSampleLayer(layer1_Upsample, layer2_Units, layer2_KernelSize, layer2_MaxpoolCount, "Layer2Up")
        layer3_Upsample = self._AddUpSampleLayer(layer2_Upsample, layer1_Units, layer1_KernelSize, layer1_MaxpoolCount, "Layer3Up")

        # 預測
        predict = conv2d(layer3_Upsample, self.OutClass, 3, 1, 'same', name="Predict")
        predictImgProb = tf.nn.softmax(predict, axis=3, name= "PredictProb")
        self.PredictImg = tf.cast(tf.reshape(tf.argmax(predictImgProb, axis=3, name="PredictImg"), [-1, self.SizeY, self.SizeX, 1]), tf.uint8) * 255

        with tf.name_scope("Loss"):
            loss = tf.losses.softmax_cross_entropy(self.LabeledData, predict)
            self.Optimzer = tf.train.AdamOptimizer(lr).minimize(loss)

        # Log
        self.LossTensor = tf.summary.scalar("Loss", loss)
        self.ExampleTensor = tf.summary.image("Example", self.PredictImg)

    # 預測
    def Train(self, DM, epochNum, batchSize):
        for i in tqdm(range(epochNum + 1)):
            # 抓取資料
            Train_BatchData, Labeled_BatchData = DM.BatchTrainData(batchSize)
            feed_dict = {
                self.InputData: Train_BatchData,
                self.LabeledData: Labeled_BatchData
            }

            # 紀錄 Train 的結果
            if i % 100 == 0:
                feed_dict_FirstN = {
                    self.InputData: DM.TestFirstBoxOfData(3),
                }
                lossSummary = self.Session.run(self.LossTensor, feed_dict=feed_dict)
                exampleSummary = self.Session.run(self.ExampleTensor, feed_dict=feed_dict_FirstN)

                # mergeSummary = tf.Summary.merge([lossSummary, exampleSummary])
                self.LogWriter.add_summary(lossSummary, i)
                self.LogWriter.add_summary(exampleSummary, i)

            # 更新
            self.Session.run(self.Optimzer, feed_dict=feed_dict)



    # 預測資料
    def Predict(self, Data):
        # Data
        feed_dict = {
            self.InputData: Data
        }
        return self.Session.run(self.PredictImg, feed_dict=feed_dict)


    # 存 Weight
    def SaveWeight(self,  logdir="./logs"):
        saver = tf.train.Saver()
        saver.save(self.Session, logdir + "/Model.ckpt")

    # 清除記憶體
    def Release(self):
        # 清除之前的 Graph
        self.Session.close()
        tf.reset_default_graph()

    # Helper Function
    def _AddConvoluationLayer(self, inpuTensor, units, kernel_size, padding, maxpoolSize, name):
        with tf.name_scope(name):
            layer_conv = conv3d(inpuTensor, units, kernel_size, padding, "same", name= name + "_Conv")
            layer_bn = batch_normalization(layer_conv, name= name +  "_BN")
            layer_act = relu(layer_bn, name= name + "_Relu")
            layer_maxpool = max_pooling3d(layer_act, maxpoolSize, maxpoolSize, "same", name= name + "_Maxpool")
            layer_dropout = dropout(layer_maxpool, 0.5, name= name + "_DropOut")
        return layer_dropout
    def _MidTo2D(self, inputTensor, units):
        with tf.name_scope("MidLayer"):
            layer_conv = conv3d(inputTensor, self.OutClass, [units, 1, 1], 1, name = "MidLayer_Conv")
            layer_reshape = tf.reshape(layer_conv, [-1, units, units, 2], name = "ReshapeTo2D")
        return layer_reshape
    def _AddUpSampleLayer(self, inputTensor, units, kernel_size, padding, name):
        with tf.name_scope(name):
            layer_upsample = conv2d_transpose(inputTensor, units, kernel_size, padding, 'same', name= name + "_Upsample")
            layer_bn = batch_normalization(layer_upsample, name= name + "_BN")
            layer_act = relu(layer_bn, name= name + "_Relu")
            layer_dropout = dropout(layer_act, 0.5, name= name + "_Dropout")
        return layer_dropout