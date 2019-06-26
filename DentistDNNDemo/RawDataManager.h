#pragma once
/*
這邊是管理所有裝置的 class (包含 藍芽、OCT)
*/
#using <System.dll>
#include "CudaV2/TRCudaV2.cuh"
#include "CudaV2/UtilityTools.cuh"

#include "DataManager.h"

#include <vcclr.h>								// 要使用 Manage 的 Class，要拿這個使用 gcroot
#include <functional>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include <QFile>
#include <QIODevice>
#include <QComboBox>
#include <QLineEdit>
#include <QTextStream>
#include <QDataStream>
#include <QLabel>
#include <QByteArray>
#include <QPixmap>
#include <QImage>
#include <QQuaternion>
#include <QVector2D>
#include <QStringList>
#include <QTemporaryDir>
#include <QProcess>
#include <QElapsedTimer>
#include <QMatrix4x4>
#include <QTextCodec>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace System::IO::Ports;

// 讀資料的部分
enum RawDataType
{
	SINGLE_DATA_TYPE = 0,
	MULTI_DATA_TYPE,
	ERROR_TYPE,
};

// Bounding Box 的 DataStruct
struct BoundingBoxDataStructRaw
{
	vector<cv::Point> contoursRaw;					// 這個是原始的邊界
	vector<cv::Point> contoursPoly;					// 這個是對輪廓做多邊形擬合之後的邊界
	Rect boundingRect;								// 框框框起來
};

class RawDataManager
{
public:
	RawDataManager();
	~RawDataManager();

	//////////////////////////////////////////////////////////////////////////
	// OCT 相關的步驟
	//
	// 底下這邊的步驟可以三選一
	// 可以：
	// ReadRawDataFromFileV2			=> 讀檔，根據檔案大小來判斷是 Single 還是 Multi 的資料
	// ScanSingleDataFromDeviceV2		=> 直接從 OCT 讀單張資料
	// ScanMultiDataFromDeviceV2		=> 直接從 OCT 讀多張資料
	//
	// 註： 底下的 Function 排序 跟 ScanningWorkerThread Call 的順序一樣
	//////////////////////////////////////////////////////////////////////////
	RawDataType ReadRawDataFromFileV2(QString);									// 有修改的過後的 Raw Data Reader
	void TransformToIMG(bool);													// 轉換成圖檔 (是否要加入邊界資訊在圖檔內)
	void TransformToOtherSideView();											// 轉出TopView 

	//////////////////////////////////////////////////////////////////////////
	// Network or Volume 相關的 Function
	//////////////////////////////////////////////////////////////////////////
	void NetworkDataGenerateInRamV2();											// 產生相同的類神經網路資料，但不輸出
	bool CheckIsValidData();													// 有的時候 Eigen 會算出有問題的解，所以
	void SaveNetworkImage();
	void LoadPredictImage();													// 將預測的圖讀進來
	void SmoothNetworkData();													// 優化 Network 預測出來的雜點
	void NetworkDataToQImage();													// 轉成 QImage

private:
	//////////////////////////////////////////////////////////////////////////
	// 其他 Class 的資料
	//////////////////////////////////////////////////////////////////////////
	DataManager			DManager;
	TRCudaV2			cudaV2;
	UtilityTools		utilityTools;

	//////////////////////////////////////////////////////////////////////////
	// 網路抓 Bounding Box
	//////////////////////////////////////////////////////////////////////////
	Mat								GetBoundingBox(Mat, QVector2D&, QVector2D&);			// 網路在抓取的時候，有一個 Bounding Box 可以減少拿到外側的機率
	static bool						SortByContourPointSize(BoundingBoxDataStructRaw&,			// 根據 Contour 的點來排序
														BoundingBoxDataStructRaw&);
	Size							BlurSize = Size(9, 9);									// 模糊的區塊
	int								BoundingThreshold = 8;									// 這邊是要根據多少去裁減
	int								BoundingOffset = 0;										// 加上 Offset

	//////////////////////////////////////////////////////////////////////////
	// 其他變數
	//////////////////////////////////////////////////////////////////////////
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");

	//////////////////////////////////////////////////////////////////////////
	// 存圖片的陣列
	//////////////////////////////////////////////////////////////////////////
	QVector<Mat>		ImageResultArray;										// 原圖
	QVector<Mat>		BorderDetectionResultArray;								// 邊界判斷完
	QVector<Mat>		NetworkResultArray;										// 網路預測的結果(0-140)
	Mat					OtherSideMat;											// 存放單一大小的

	//////////////////////////////////////////////////////////////////////////
	// 顯示部分
	//////////////////////////////////////////////////////////////////////////
	QVector<QImage>		QImageResultArray;										// 同上(顯示)
	QVector<QImage>		QBorderDetectionResultArray;							// 同上(顯示)
	QVector<QImage>		QNetworkResultArray;									// 同上(顯示)

	//////////////////////////////////////////////////////////////////////////
	// 網路的 Bounding Point
	//////////////////////////////////////////////////////////////////////////
	QVector<QVector2D>	TLPointArray;											// Bounding Box Array
	QVector<QVector2D>	BRPointArray;											// 同上

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	QImage				Mat2QImage(Mat const &, int);
	int					clamp(int, int, int);
};