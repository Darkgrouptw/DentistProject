#include "TensorflowNet.h"

#include <QString>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

// Helper Function
float** MatToFloatArray(Mat img)
{
	float* _data = new float[img.rows * img.cols];
	float** data = new float*[img.rows];
	for (int i = 0; i < img.rows; i++)
	{
		data[i] = &_data[i * img.cols];

		#pragma omp parallel for
		for (int j = 0; j < img.cols; j++)
		{
			float pixel = (float)img.at<uchar>(i, j);
			data[i][j] = pixel;
		}
	}
	return data;
}
uchar* FloatArrayToUchar(float** img, int rows, int cols)
{
	uchar* ImgData = new uchar[rows * cols];
	#pragma omp parallel for
	for (int i = 0; i < rows * cols; i++)
	{
		int rowIndex = i / cols;
		int colIndex = i % cols;
		float pointData = img[rowIndex][colIndex];
		if (pointData <= 0)
			pointData = 0;
		else if (pointData >= 255)
			pointData = 255;
		ImgData[i] = (uchar)(pointData);
	}
	return ImgData;
}

int main(int argc, char *argv[])
{
	#pragma region 先判斷是否能使用
	if (argc != 4)
	{
		cout << "檔案路徑沒有給!!" << endl;
		return -1;
	}
	#pragma endregion
	QString mode = QString(argv[3]);
	if (mode == "OtherSide")
	{
		#pragma region 顯示抓到的結果
		QString ImgPath(argv[1]);
		cout << "ImgPath: " << ImgPath.toLocal8Bit().toStdString() << endl;
		#pragma endregion
		#pragma region 圖片處理
		Mat Img = imread(ImgPath.toLocal8Bit().toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
		float** ImgData = MatToFloatArray(Img);

		TensorflowNet net;
		float** PredictData = net.Predict_OtherSide(ImgData);
		uchar* PredictDataChar = FloatArrayToUchar(PredictData, 250, 250);

		Mat PredictImg(250, 250, CV_8UC1, PredictDataChar, cv::Mat::AUTO_STEP);
		cout << "CCC: " << (int)PredictImg.at<uchar>(0, 0) << endl;
		QString PredictImgPath(argv[2]);
		imwrite(PredictImgPath.toLocal8Bit().toStdString(), PredictImg);
		#pragma endregion
	}
	else if (mode == "Full")
	{
		#pragma region 抓取檔案結果
		QString ImgPath(argv[1]);
		cout << "ImgPath: " << ImgPath.toLocal8Bit().toStdString() << endl;
		#pragma endregion
	}	
	return 0;
}
