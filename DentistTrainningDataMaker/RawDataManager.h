#pragma once
/*
這邊是管理所有裝置的 class (包含 藍芽、OCT)
*/
#include "DataManager.h"
#include "TRCuda.cuh"
#include "BluetoothManager.h"

#include <QVector>
#include <QDataStream>
#include <QByteArray>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// 這邊是為了要讓邊界 Smooth 一點
struct IndexMapInfo
{
	int index;			// 位置資訊
	int ZValue;			// Z 的深度值是多少
};

class RawDataManager
{
public:
	RawDataManager();
	~RawDataManager();

	//////////////////////////////////////////////////////////////////////////
	// 底下這邊的步驟可以二選一
	// 可以：
	// ReadRawDataFromFile	=> 讀檔，然後把資料存起來
	// ScanDataFromDevice	=> 直接從 OCT 讀資料
	//////////////////////////////////////////////////////////////////////////
	void ReadRawDataFromFile(QString);
	void ScanDataFromDevice(QString);
	void RawToPointCloud();
	void TranformToIMG(bool);

	BluetoothManager	bleManager;
private:
	// 以前的資料
	DataManager			DManager;
	TRcuda				theTRcuda;
	cv::Mat				OCTMask;

	//////////////////////////////////////////////////////////////////////////
	// OCT
	//////////////////////////////////////////////////////////////////////////
	string				OCTDevicePort = "COM6";									// 這個是那台機器預設的 COM 位置
	unsigned int		OCT_HandleOut;
	unsigned int		OCT_DataLen;
	unsigned int		OCT_AllDataLen;
	bool				OCT_ErrorBoolean;
	int					OCT_DeviceID;

	// 存圖片的陣列
	QVector<cv::Mat>	ImageResultArray;
	QVector<cv::Mat>	CutFFTBorderArray;
	QVector<cv::Mat>	FastLabelArray;
	QVector<cv::Mat>	CombineTestArray;

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	int					LerpFunction(int, int, int, int, int);

	QByteArray buffer;
};

