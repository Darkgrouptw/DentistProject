#pragma once
/*
這邊是管理所有裝置的 class (包含 藍芽、OCT)
*/
#include "DataManager.h"
#include "TRCuda.cuh"
#include "BluetoothManager.h"

#include <cmath>
#include <vector>
#include <QVector>
#include <QDataStream>
#include <QLabel>
#include <QByteArray>
#include <QPixmap>
#include <QImage>
#include <QMessageBox>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

// 這邊是為了要讓邊界 Smooth 一點
/*struct IndexMapInfo
{
	int index;			// 位置資訊
	int ZValue;			// Z 的深度值是多少
};*/

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
	// OCT 相關的步驟
	//
	// 底下這邊的步驟可以二選一
	// 可以：
	// ReadRawDataFromFile	=> 讀檔，然後把資料存起來
	// ScanDataFromDevice	=> 直接從 OCT 讀資料
	//////////////////////////////////////////////////////////////////////////
	void ReadRawDataFromFile(QString);
	void ScanDataFromDevice(QString);											// 輸入儲存路徑
	void RawToPointCloud();														// 把 Raw 檔，轉乘點雲
	void TranformToIMG(bool);													// 轉換成圖檔
	bool ShakeDetect(QMainWindow*, bool);										// 

	// 藍芽的部分
	BluetoothManager	bleManager;
private:
	// 以前的資料
	DataManager			DManager;
	TRcuda				theTRcuda;


	//////////////////////////////////////////////////////////////////////////
	// OCT
	//////////////////////////////////////////////////////////////////////////
	string				OCTDevicePort = "COM6";									// 這個是那台機器預設的 COM 位置
	unsigned int		OCT_HandleOut;
	unsigned int		OCT_DataLen;
	unsigned int		OCT_AllDataLen;
	bool				OCT_ErrorBoolean;
	int					OCT_DeviceID;

	//////////////////////////////////////////////////////////////////////////
	// 存圖片的陣列
	//////////////////////////////////////////////////////////////////////////
	QVector<cv::Mat>	ImageResultArray;
	QVector<cv::Mat>	SmoothResultArray;
	//QVector<cv::Mat>	FastLabelArray;
	QVector<cv::Mat>	CombineTestArray;

	//////////////////////////////////////////////////////////////////////////
	// UI Pointer
	//////////////////////////////////////////////////////////////////////////
	QLabel*				ImageResult;
	QLabel*				FinalResult;

	//////////////////////////////////////////////////////////////////////////
	// 儲存設定
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	int					LerpFunction(int, int, int, int, int);
	QImage				Mat2QImage(cv::Mat const &, int);
	vector<cv::DMatch>	SURF_Feature_Detect(cv::Mat, cv::Mat, bool IsShowDebug = false);							// 做 SIFT 判斷


	QByteArray buffer;
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");
};

