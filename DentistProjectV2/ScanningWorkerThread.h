#pragma once
/*
這邊是管理 System::Thread 的 class
由於這邊是使用 clr (Manage 的物件)
所以class 要用 public ref class (因為 clr 不支援 c++ thread & QThread)
*/
#include "GlobalDefine.h"

#include <iostream>
#include <vector>
#include <functional>

#include <QFile>
#include <QDir>
#include <QTime>
#include <QSlider>
#include <QLineEdit>
#include <QString>
#include <QPushButton>
#include <QMatrix4x4>

using namespace std;
using namespace System::Threading;

// 這個是用來判斷是否有掃描過
struct PointTypeInfo {
	bool IsEmpty = true;				// 如果前面有掃描過，就要改 true
	int* TypeData = NULL;				// 掃描完之後，要 Copy 資料
};

public ref class ScanningWorkerThread
{
public:
	ScanningWorkerThread(int);
	~ScanningWorkerThread();

	//////////////////////////////////////////////////////////////////////////
	// 傳送 Function Pointer
	//////////////////////////////////////////////////////////////////////////
	void InitScanFunctionPointer(												// 初始化 Scan 的 Function Pointer
		function<void(QString, bool)>*,											// Single
		function<void(QString, bool)>*,											// Multi
		function<void(bool)>*,													// ToImage
		function<void()>*,														// ToOtherSide
		function<QQuaternion()>*												// QQuaternion
	);
	void IntitShakeDetectFunctionPointer(
		function<void(int*&)>*,													// Copy 單張資訊
		function<bool(int*, bool)>*,											// Shake Detect (Single)
		function<bool(bool)>*													// Shake Detect (Multi)
	);
	void InitShowFunctionPointer(
		function<void(QQuaternion)>*,											// Save 點雲
		function<void()>*,														// Align 點雲
		function<void()>*														// ShowImageIndex
	);
	void InitUIPointer(QSlider*, QPushButton*, QLineEdit*);						// 初始化 UI Pointer (Thread 中，做完會去改圖，所以需要這個 Function Pointer)

	//////////////////////////////////////////////////////////////////////////
	// 外部掃描的 Function
	//////////////////////////////////////////////////////////////////////////
	void SetParams(QString*, bool, bool, bool, bool);							// 設定參數
	void SetScanModel(bool);													// 設定掃描 Mode & 設定是否儲存檔案

private:
	//////////////////////////////////////////////////////////////////////////
	// 掃描相關
	//
	// Thread 的掃描狀態:
	// 1. 開始掃描，要顯示畫面
	// 2. 掃描單張資料
	// 3. 等待結果判斷
	// 4. 如果是一張，那還要再掃一張，如果大於一張，驗證最後兩張是否正確
	// 5. 開始掃描多個資料
	// 6. 等待結果判斷
	// 7. 驗證是否有晃動到
	// 8. 如果沒有晃到，那就儲存點雲 & 如果大於二就執行拼接
	//////////////////////////////////////////////////////////////////////////
	void ScanProcess();															// Thread 跑的 Function
	bool IsEnd = true;															// 是否要結束
	bool NeedSave_Single_RawData = false;										// 是否要儲存 Single Raw Data
	bool NeedSave_Multi_RawData = false;										// 是否要儲存 Multi Raw Data
	bool NeedSave_ImageData = false;											// 是否要儲存完 Image Data
	bool AutoDelete_ShakeData = false;											// 是否要自動刪除晃動資料
	PointTypeInfo* Last_PointType_1D = NULL;									// 掃描玩的時候需要去抓上一張的結果

	//////////////////////////////////////////////////////////////////////////
	// UI 設定 (由於變數名稱是比照 UI 的變數，所以就沒有註解，就是直接對應)
	//////////////////////////////////////////////////////////////////////////
	QSlider* ScanNumSlider;
	QLineEdit* SaveLocationText;
	QString* EndScanText;
	QPushButton* ScanButton;

	//////////////////////////////////////////////////////////////////////////
	// 九軸資料
	//////////////////////////////////////////////////////////////////////////
	QFile* GyroFile;
	QVector<QQuaternion>* quatList;

	//////////////////////////////////////////////////////////////////////////
	// 實際在執行的 Thread
	//////////////////////////////////////////////////////////////////////////
	Thread^ ScanThread;

	//////////////////////////////////////////////////////////////////////////
	// Function Pointer
	//////////////////////////////////////////////////////////////////////////
	function<void(QString, bool)>*	ScanSingleDataFromDeviceV2 = NULL;			// 掃描單張資料
	function<void(QString, bool)>*	ScanMultiDataFromDeviceV2 = NULL;			// 掃描多張資料
	function<void(bool)>*			TransformToIMG = NULL;						// 將資料轉成圖 (準備顯示前)
	function<void()>*				TransformToOtherSideView = NULL;			// TopView 
	function<QQuaternion()>*		GetQuaternionFromDevice = NULL;				// 從裝置拿旋轉量
	function<void(int*&)>*			CopySingleBorder = NULL;					// 抓出單張資訊
	function<bool(int*, bool)>*		ShakeDetect_Single = NULL;					// 是否有晃動 (Single)
	function<bool(bool)>*			ShakeDetect_Multi = NULL;					// 是否有晃動 (Multi)
	function<void(QQuaternion)>*	SavePointCloud = NULL;						// 因為這邊不用做比對，所以直接把點雲存出來顯示就可以了
	function<void()>*				AlignmentPointCloud = NULL;					// 拼接點雲的 Function
	function<void()>*				ShowImageIndex = NULL;						// 顯示在畫面上
};

