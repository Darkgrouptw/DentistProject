﻿#pragma once
/*
這邊是管理所有裝置的 class (包含 藍芽、OCT)
*/
#include "DataManager.h"
#include "TRCuda.cuh"
#include "CudaBorder.cuh"
#include "BluetoothManager.h"
#include "PointCloudInfo.h"

#include "4pcs.h"
#include "super4pcs/shared4pcs.h"
#include "super4pcs/algorithms/super4pcs.h"
#include "super4pcs/io/io.h"

#include <cmath>
#include <vector>

#include <QFile>
#include <QIODevice>
#include <QTextStream>
#include <QDataStream>
#include <QLabel>
#include <QByteArray>
#include <QPixmap>
#include <QImage>
#include <QMessageBox>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct TransformVisitor {
	inline void operator() (
		float fraction,
		float best_LCP,
		Eigen::Ref<GlobalRegistration::Match4PCSBase::MatrixType> /*transformation*/) {
		printf("done: %d%c best: %f                  \r",
			static_cast<int>(fraction * 100), '%', best_LCP);
		fflush(stdout);
	}
	constexpr bool needsGlobalTransformation() const { return false; }
};

//struct PointCloudInfo {

//};

class RawDataManager
{
public:
	RawDataManager();
	~RawDataManager();

	//////////////////////////////////////////////////////////////////////////
	// UI 相關
	//////////////////////////////////////////////////////////////////////////
	void SendUIPointer(QVector<QObject*>);
	void ShowImageIndex(int);

	//////////////////////////////////////////////////////////////////////////
	// 九軸 or 點雲 or Alignment 相關
	//////////////////////////////////////////////////////////////////////////
	void ReadPointCloudData(QString);

	//////////////////////////////////////////////////////////////////////////
	// OCT 相關的步驟
	//
	// 底下這邊的步驟可以二選一
	// 可以：
	// ReadRawDataFromFile	=> 讀檔，然後把資料存起來
	// ScanDataFromDevice	=> 直接從 OCT 讀資料
	//////////////////////////////////////////////////////////////////////////
	void ReadRawDataFromFile(QString);											// 讀 Raw Data
	void ReadRawDataFromFileV2(QString);										// 有修改的過後的 Raw Data Reader
	void ScanDataFromDevice(QString, bool);										// 輸入儲存路徑 和 要步要儲存，來轉點雲
	void TranformToIMG(bool);													// 轉換成圖檔，轉點雲
	bool ShakeDetect(QMainWindow*, bool);										// 偵測有無晃動
	void WriteRawDataToFile(QString);											// 將 Raw Data 轉成檔案

	//////////////////////////////////////////////////////////////////////////
	// Netowrk 相關的 Function
	//////////////////////////////////////////////////////////////////////////
	QVector<cv::Mat>	GenerateNetworkData();									// 這邊是產生要預測的資料
	void				SetPredictData(QVector<cv::Mat>);						// 設定 網路預測出來的資料

	//////////////////////////////////////////////////////////////////////////
	// 點雲資料
	//////////////////////////////////////////////////////////////////////////
	QVector<PointCloudInfo> PointCloudArray;									// 每次掃描都會把結果船進去
	int					SelectIndex = 0;										// 目前選擇地的片數

	//////////////////////////////////////////////////////////////////////////
	// 藍芽的部分
	//////////////////////////////////////////////////////////////////////////
	BluetoothManager	bleManager;

private:
	// 以前的資料
	DataManager			DManager;
	TRcuda				theTRcuda;
	CudaBorder			cudaBorder;

	//////////////////////////////////////////////////////////////////////////
	// OCT
	//////////////////////////////////////////////////////////////////////////
	string				OCTDevicePort = "COM6";									// 這個是那台機器預設的 COM 位置
	unsigned int		OCT_HandleOut;
	unsigned int		OCT_DataLen;
	unsigned int		OCT_AllDataLen;
	bool				OCT_ErrorBoolean;
	int					OCT_DeviceID;
	const int			OCT_PIC_SIZE = 2048 * 2 * 500;
	const int			OCT_Shake_Pixel_Threshold = 30;							// 如果晃動大於這個 Threshold 又代表有換動到
	const float			OCT_Part_Threshold = 0.8f;								// 有效區域

	//////////////////////////////////////////////////////////////////////////
	// 網路
	//////////////////////////////////////////////////////////////////////////
	const int			NetworkCutRow = 50;
	const int			NetworkCutCol = 500;

	//////////////////////////////////////////////////////////////////////////
	// 存圖片的陣列
	//////////////////////////////////////////////////////////////////////////
	QVector<Mat>		ImageResultArray;										// 原圖								(SegNet 使用)
	QVector<Mat>		SmoothResultArray;										// Smooth 過後的結果				(邊界判斷使用)
	QVector<Mat>		CombineResultArray;										// 判斷完的結果圖					(顯示使用)

	//////////////////////////////////////////////////////////////////////////
	// 顯示部分
	//////////////////////////////////////////////////////////////////////////
	QVector<QImage>		QImageResultArray;										// 同上(顯示)
	QVector<QImage>		QSmoothResultArray;										// 同上(顯示)
	QVector<QImage>		QCombineResultArray;									// 同上(顯示)
	
	//////////////////////////////////////////////////////////////////////////
	// UI Pointer
	//////////////////////////////////////////////////////////////////////////
	QLabel*				ImageResult;											// 外部的原圖 UI Pointer
	QLabel*				NetworkResult;											// 同上，但目前是沒有用
	QLabel*				FinalResult;											// 最後找出來的結果圖

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	int					LerpFunction(int, int, int, int, int);
	QImage				Mat2QImage(cv::Mat const &, int);
	string				MarshalString(System::String^);							// 這邊跟 藍芽 Function裡面做的一樣，只是不想開 public
	void				OCT_DataType_Transfrom(unsigned short *, int , char *);	// 這邊是因為他要轉到 char
	vector<GlobalRegistration::Point3D>	ConvertQVector2Point3D(QVector<QVector3D>);	// 轉換
	void				super4PCS_Align(vector<GlobalRegistration::Point3D>*, vector<GlobalRegistration::Point3D> *, int);	// Alignment

	
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");
};

