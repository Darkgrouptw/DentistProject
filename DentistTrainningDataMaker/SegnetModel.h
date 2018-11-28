#pragma once
/*
預測網路的 Code
由於是不同於學長們寫的
所以獨立出來
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>

using namespace caffe;
using namespace cv;
using namespace std;

// 測試程式有沒有問題的型態
enum DebugMatType
{
	Type8U = 0,
	Type8UC3,
	Type32F,
	Type32FC3
};

struct NetworkSize {
	int Width;																						// col	
	int Height;																						// row
	int NumChannels;																				// channel

	NetworkSize()
	{
		this->Width = 0;
		this->Height = 0;
		this->NumChannels = 0;
	}

	NetworkSize(int w, int h, int c)
	{
		this->Width = w;
		this->Height = h;
		this->NumChannels = c;
	}

	// 轉成 CV 的 Size
	Size cvSize()
	{
		return Size(Width, Height);
	}
};

class SegNetModel
{
public:
	SegNetModel();
	~SegNetModel();

	void						Load(string, string);												// 載入架構檔 & 權重 (prototxt, caffemodel)
	void						ReshapeToMultiBatch(int);											// 改成 Multi Batch 的方式 輸入
	Mat							Predict(Mat&);														// 預測 (單張圖片)
	vector<Mat>					Predict(vector<Mat>);												// 傳入一連串的結果，預測
	Mat							Visualization(Mat);													// 轉成看得懂的圖 Output

private:
	//////////////////////////////////////////////////////////////////////////
	// 網路相關變數
	//////////////////////////////////////////////////////////////////////////
	caffe::Net<float> 			*SegNet = NULL;
	NetworkSize					InputSize;
	NetworkSize					OutputSize;
	vector<Mat>					InputChannelPointer;												// 這邊存放資料 Network 對應到每一個 Channel 的位置 Array
	string						LUT_file = "./SegNetModel/camvid11.png";
	Mat							LabelImg;
	int							BatchSize = 0;														// 是否是依次輸入多個資料
	string						ModelDefPath;														// 這邊是用在如果要重複預測 (因為 Multi Batch 的顯卡記憶體沒有辦法藉由內部的 Function 關閉)，所以會重 Load
	string						ModelPath;															// 同上

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	void						WrapSingleInputLayer();												// 指定資料到 CPU
	void						WrapMultiInputLayer();												// 同上，但是多個 Input
	void						SinglePreprocess(Mat& img);											// 轉成正確規格的圖片 (如果圖片不符合格式的話)
	void						MultiPreprocess(vector<Mat>);										// 同上，但變成多個 Input
	int							ArgMax(vector<float>);												// 找最大的值是多少

	// Debug 用的
	void						DebugMat(Mat, DebugMatType);
};