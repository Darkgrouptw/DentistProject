#include "RawDataManager.h"

RawDataManager::RawDataManager()
{
	cout << "OpenCV Version: " << CV_VERSION << endl;
	DManager.ReadCalibrationData();
}
RawDataManager::~RawDataManager()
{
}

// UI 相關
void RawDataManager::SendUIPointer(QVector<QObject*> UIPointer)
{
	// 確認是不是有多傳，忘了改的
	assert(UIPointer.size() == 3);
	ImageResult			= (QLabel*)UIPointer[0];
	NetworkResult		= (QLabel*)UIPointer[1];
	FinalResult			= (QLabel*)UIPointer[2];
}
void RawDataManager::ShowImageIndex(int index)
{
	// 這邊有兩種狀況
	// 1. 第一種狀況是 60 ~ 200
	// 2. 第二種狀況是 只有掃描一張的結果
	if (60 <= index && index <= 200 && QImageResultArray.size() > 0)
	{
		//QImage Pixmap_ImageResult = QImageResultArray[index];
		//ImageResult->setPixmap(QPixmap::fromImage(Pixmap_ImageResult));

		////QImage Pixmap_NetworkResult = Mat2QImage(FastBorderResultArray[index - 60], CV_8UC3);
		////NetworkResult->setPixmap(QPixmap::fromImage(Pixmap_NetworkResult));

		//// 如果有東西的話才顯示 Network 預測的結果
		//if (QCombineResultArray.size() > 0)
		//{
		//	QImage Pixmap_FinalResult = QCombineResultArray[index];
		//	FinalResult->setPixmap(QPixmap::fromImage(Pixmap_FinalResult));
		//}
	}
}

// OCT 相關的步驟

