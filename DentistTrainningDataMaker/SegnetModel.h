#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include <chrono>

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

class SegnetModel
{
public:
	SegnetModel();
	~SegnetModel();

	void						Load(string, string);
	Mat							Predict(Mat&);
	Mat							Visualization(Mat);													// 轉成看得懂的圖 Output

private:
	//////////////////////////////////////////////////////////////////////////
	// 網路相關變數
	//////////////////////////////////////////////////////////////////////////
	caffe::Net<float> 			*SegNet;
	NetworkSize					InputSize;
	NetworkSize					OutputSize;
	vector<Mat>					InputChannelPointer;												// 這邊存放資料 Network 對應到每一個 Channel 的位置 Array
	string						LUT_file = "./Models/camvid11.png";
	Mat							LabelImg;

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	void						WrapInputLayer();													// 用來轉圖片用的 (如果圖片不符合格式的話)
	void						Preprocess(Mat& img);												// 同樣，轉成正確規格的圖片
	int							ArgMax(vector<float>);												// 找最大的值是多少

	// Debug 用的
	void						DebugMat(Mat, DebugMatType);
};