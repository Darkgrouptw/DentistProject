#pragma once
/*
這邊是管理所有裝置的 class (包含 藍芽、OCT)
*/
#include "TRCudaV2.cuh"
#include "TensorflowNet_OtherSide.h"

#include "DataManager.h"
#include "BluetoothManager.h"
#include "PointCloudInfo.h"
#include "VolumeRenderClass.h"
#include "ScanningWorkerThread.h"

#include "4pcs.h"
#include "super4pcs/shared4pcs.h"
#include "super4pcs/algorithms/super4pcs.h"
#include "super4pcs/io/io.h"

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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// 讀資料的部分
enum RawDataType
{
	SINGLE_DATA_TYPE = 0,
	MULTI_DATA_TYPE,
	ERROR_TYPE,
};

// 4PCS 的 Data Structor
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

// Bounding Box 的 DataStruct
struct BoundingBoxDataStruct
{
	vector<Point> contoursRaw;					// 這個是原始的邊界
	vector<Point> contoursPoly;					// 這個是對輪廓做多邊形擬合之後的邊界
	Rect boundingRect;							// 框框框起來
};

using namespace GlobalRegistration;

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

	//void ReadPointCloudData(QString);

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
	void ScanSingleDataFromDeviceV2(QString, bool);								// 輸入儲存路徑 和 要步要儲存
	void ScanMultiDataFromDeviceV2(QString, bool);								// 輸入儲存路徑 和 要步要儲存
	void TransformToIMG(bool);													// 轉換成圖檔 (是否要加入邊界資訊在圖檔內)
	void TransformToOtherSideView();											// 轉出TopView 
	QQuaternion GetQuaternion();												// 從藍芽中拿資料出來
	void SetScanOCTMode(bool, QString*, bool, bool, bool, bool);				// 開始掃描 OCT
	void SetScanOCTOnceMode();													// 只掃描一張
	void CopySingleBorder(int *&);												// 存單張 Border
	bool ShakeDetect_Single(int *, bool);										// 有無晃動 (單)
	bool ShakeDetect_Multi(bool, bool);											// 有無晃動 (多)
	void SavePointCloud(QQuaternion);											// 因為這邊不用做比對，所以直接把點雲存出來顯示就可以了
	void AlignmentPointCloud();													// 跟以前的點雲資料做對齊
	void CombinePointCloud(int, int);											// 合併點雲	(後面加到前面)

	//////////////////////////////////////////////////////////////////////////
	// Network or Volume 相關的 Function
	//////////////////////////////////////////////////////////////////////////
	void NetworkDataGenerateV2(QString);										// 產生類神經網路的資料
	void NetworkDataGenerateInRamV2();											// 產生相同的類神經網路資料，但不輸出
	//void ImportVolumeDataTest(QString);										// 輸入 Label 資料做測試用
	//QVector<VolumeRenderClass*> VolumeDataArray;								// Network 預測完的資料

	//////////////////////////////////////////////////////////////////////////
	// 點雲資料
	//////////////////////////////////////////////////////////////////////////
	QVector<PointCloudInfo> PointCloudArray;									// 每次掃描都會把結果船進去
	int					SelectIndex = -1;										// 目前選擇地的片數
	bool				IsLockPC = false;										// Lock PC 是來判斷是否有新資料，有新資料就會 Lock，更新結束，就會取消 Lock
	bool				IsLockVolumeData = false;								// 同上，可能後面會取代
	bool				IsWidgetUpdate = false;									// 是否有介面在更新
	QVector<QQuaternion> QuaternionList;										// 修改用		
	void				PCWidgetUpdate();										// 更新點部分的資料
	void				RotationAngle(int);										// 更新旋轉的角度
	void				TransformMultiDataToPointCloud(QStringList);			// 將資料轉成點雲

	//////////////////////////////////////////////////////////////////////////
	// 藍芽的部分
	//////////////////////////////////////////////////////////////////////////
	BluetoothManager	bleManager;

private:
	//////////////////////////////////////////////////////////////////////////
	// 其他 Class 的資料
	//////////////////////////////////////////////////////////////////////////
	DataManager			DManager;
	TRCudaV2			cudaV2;
	gcroot<ScanningWorkerThread^> Worker;

	//////////////////////////////////////////////////////////////////////////
	// OCT
	//////////////////////////////////////////////////////////////////////////
	string				OCTDevicePort = "COM6";									// 這個是那台機器預設的 COM 位置
	unsigned int		OCT_HandleOut;
	unsigned int		OCT_DataLen;
	unsigned int		OCT_AllDataLen;
	bool				OCT_ErrorBoolean;
	int					OCT_DeviceID;
	const int			OCT_PIC_SIZE = 2048 * 250 * 2 * 2;						// 2 (來回) & 2 Channel
	QString				OCT_SUCCESS_TEXT = "ATS : ApiSuccess (512)";			// 正常的話，會顯示這個 Message

	//////////////////////////////////////////////////////////////////////////
	// Function Pointer
	//////////////////////////////////////////////////////////////////////////
	function<void(QString, bool)>	ScanSingle_Pointer;
	function<void(QString, bool)>	ScanMulti_Pointer;
	function<void(bool)>			TransforImage_Pointer;
	function<void()>				TransformToOtherSideView_Pointer;
	function<QQuaternion()>			GetQuaternion_Pointer;
	function<void(int*&)>			CopySingleBorder_Pointer;
	function<bool(int*, bool)>		ShakeDetect_Single_Pointer;
	function<bool(bool, bool)>		ShakeDetect_Multi_Pointer;
	function<void(QQuaternion)>		SavePointCloud_Pointer;
	function<void()>				AlignmentPointCloud_Pointer;
	function<void()>				ShowImageIndex_Pointer;

	//////////////////////////////////////////////////////////////////////////
	// 網路
	//////////////////////////////////////////////////////////////////////////
	Mat					GetBoundingBox(Mat, QVector2D&, QVector2D&);			// 網路在抓取的時候，有一個 Bounding Box 可以減少拿到外側的機率
	static bool			SortByContourPointSize(BoundingBoxDataStruct&,			// 根據 Contour 的點來排序
											BoundingBoxDataStruct&);
	Size				BlurSize = Size(9, 9);									// 模糊的區塊
	int					BoundingThreshold = 8;									// 這邊是要根據多少去裁減
	int					BoundingOffset = 0;										// 加上 Offset
	//QVector<>

	//////////////////////////////////////////////////////////////////////////
	// 4PCS 常數
	//////////////////////////////////////////////////////////////////////////
	const float			AlignScoreThreshold = 0.2f;

	//////////////////////////////////////////////////////////////////////////
	// 存圖片的陣列
	//////////////////////////////////////////////////////////////////////////
	QVector<Mat>		ImageResultArray;										// 原圖
	QVector<Mat>		BorderDetectionResultArray;								// 邊界判斷完
	QVector<Mat>		NetworkResultArray;										// 網路預測的結果
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
	// UI Pointer
	//////////////////////////////////////////////////////////////////////////
	QLabel*				ImageResult;											// 外部的原圖 UI Pointer
	QLabel*				BorderDetectionResult;									// 最後找出來的結果圖
	QLabel*				NetworkResult;											// 同上，但目前是沒有用
	QLabel*				OtherSideResult;										// TopView 
	QObject*			DisplayPanel;											// 畫圖的部分
	QComboBox*			PCIndex;												// 選擇 PC 的 Index

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	int					LerpFunction(int, int, int, int, int);
	QImage				Mat2QImage(Mat const &, int);
	string				MarshalString(System::String^);							// 這邊跟 藍芽 Function裡面做的一樣，只是不想開 public
	void				OCT_DataType_Transfrom(unsigned short *, int, char *);	// 這邊是因為他要轉到 char
	void				ConvertQVector2Point3D(QVector<QVector3D>&, vector<Point3D>&);	// 同上
	void				ConvertPoint3D2QVector(vector<Point3D>&, QVector<QVector3D>&);	// 同上
	QMatrix4x4			super4PCS_Align(vector<Point3D>*, vector<Point3D> *, float&);	// Alignment

	//////////////////////////////////////////////////////////////////////////
	// TransformMatrix
	//////////////////////////////////////////////////////////////////////////
	static const int	CONST_SIZE_MATRIX = 12 + 1;								// 矩陣大小			
	QString		PreMolarLoc = "./AlignmentMatrix/PreMolar.txt";					// Alignment 的位置
	QMatrix4x4	PreMolarMatrix[CONST_SIZE_MATRIX];								// 矩陣

	//////////////////////////////////////////////////////////////////////////
	// 其他變數
	//////////////////////////////////////////////////////////////////////////
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");
	QVector3D PanelPointOffset = QVector3D(0, 0, 0);
};