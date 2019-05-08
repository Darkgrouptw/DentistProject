#include <Windows.h>

#include "TRCudaV2.cuh"

#include <iostream>

#include <QDataStream>
#include <QTextStream>
#include <QFile>
#include <QString>

using namespace std;

int main(int argc, char *argv[])
{
	#pragma region 例外狀況排除
	// Windows 10 1809 的 bug
	system("chcp 65001");
	system("chcp 950");

	if (argc != 2)
	{
		cout << "檔名錯誤" << endl;
		return -1;
	}

	QString FileName = QString(argv[1]);
	QFile inputFile(FileName);
	cout << FileName.toLocal8Bit().constData() << endl;
	if (!inputFile.open(QIODevice::ReadOnly))
	{
		cout << "Raw Data 讀取錯誤" << endl;
		return -1;
	}
	else
		cout << "讀取 Raw Data: " << FileName.toLocal8Bit().constData() << endl;

	#pragma endregion
	#pragma region 抓取全部的 Bytes
	int bufferSize = inputFile.size() / sizeof(quint8);

	QDataStream qData(&inputFile);
	QByteArray buffer;
	buffer.clear();
	buffer.resize(bufferSize);
	int status = qData.readRawData(buffer.data(), bufferSize);

	//-1 代表有錯誤
	assert(status != -1);

	inputFile.close();
	#pragma endregion
	#pragma region 開始做轉換
	// 讀資料 (後面的參數是東元那邊測試出來的 )
	TRCudaV2 cudaV2;

	// 測試 Single Scan
	//cudaV2.SingleRawDataToPointCloud(buffer.data(), bufferSize, 250, 2048, 37 * 4 - 4, 2, 10);

	// 測試 Multi Scan
	cudaV2.MultiRawDataToPointCloud(buffer.data(), bufferSize, 250, 250, 2048, 37 * 4 - 4, 2, 10);

	// 晃動判斷
	//cudaV2.ShakeDetect(false);
	#pragma endregion
	#pragma region 測試圖片
	//vector<Mat> ImgArray = cudaV2.TransfromMatArray(true);
	vector<Mat> ImgArray = cudaV2.TransfromMatArray(false);

	//for (int x = 0; x < 250; x++)
	//	imwrite("Images/Origin/" + to_string(x) + ".png", ImgArray[x]);

	// 判斷是否要檢查所有的 Min Max 點
	int TestBestNum = 3;
	for (int x = 0; x < 250; x++)
	{
		Mat PeakImg = ImgArray[x].clone();
		for (int row = 0; row < PeakImg.rows; row++)
		{
			for (int bestIndex = 0; bestIndex < TestBestNum; bestIndex ++)
			{
				if(cudaV2.TestBestN[x * PeakImg.rows * TestBestNum + row * TestBestNum + bestIndex] == -1)
					continue;
				Point contourPoint(cudaV2.TestBestN[x * PeakImg.rows * TestBestNum + row * TestBestNum + bestIndex], row);
				circle(PeakImg, contourPoint, 1, Scalar(0, 255, 255), CV_FILLED);
			}
		}
		imwrite("Images/Peak/" + to_string(x) + ".png", PeakImg);
	}
	// imwrite("Images/" + to_string(x) + ".png", ImgArray[x]);

	// 多張圖片
	ImgArray = cudaV2.TransfromMatArray(true);
	for (int x = 0; x < 250; x++)
		imwrite("Images/Border/" + to_string(x) + ".png", ImgArray[x]);
	#pragma endregion
	system("PAUSE");
	return 0;
}
// "E:/DentistData/ScanData/2019.01.08_ToothBone/DATA/TOOTH bone 2"
// "E:/DentistData/ScanData/2019.01.08_ToothBone/DATA/TOOTH bone 3.1"
// "E:/DentistData/ScanData/2019.03.05_ToothBone/5_slim"
// "E:/DentistData/ScanData/2019.04.15_ToothBone/DONG31"