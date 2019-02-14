#include "ScanningWorkerThread.h"

ScanningWorkerThread::ScanningWorkerThread(int rows)
{
	// 初始化變數
	Last_PointType_1D = new PointTypeInfo;
	Last_PointType_1D->IsEmpty = true;
	Last_PointType_1D->TypeData = new int[rows];
}
ScanningWorkerThread::~ScanningWorkerThread()
{
	delete[] Last_PointType_1D->TypeData;
	delete Last_PointType_1D;
}

// 傳送 Function Pointer
void ScanningWorkerThread::InitScanFunctionPointer(
	function<void(QString, bool)>* ScanSingle,
	function<void(QString, bool)>* ScanMulti,
	function<void(bool)>* ToImage,
	function<QQuaternion()>* Quaternion)
{
	ScanSingleDataFromDeviceV2 = ScanSingle;
	ScanMultiDataFromDeviceV2 = ScanMulti;
	TransformToIMG = ToImage;
	GetQuaternionFromDevice = Quaternion;
}
void ScanningWorkerThread::IntitShakeDetectFunctionPointer(
	function<void(int*&)>* CopyBorderInfo,
	function<bool(int*, bool)>* ShakeDetectSingle,
	function<bool(bool)>* ShakeDetectMulti)
{
	CopySingleBorder = CopyBorderInfo;
	ShakeDetect_Single = ShakeDetectSingle;
	ShakeDetect_Multi = ShakeDetectMulti;
}
void ScanningWorkerThread::InitShowFunctionPointer(
	function<void(QQuaternion)>* SavePC,
	function<void()>* AlignPC,
	function<void()>* showImage)
{
	SavePointCloud = SavePC;
	AlignmentPointCloud = AlignPC;
	ShowImageIndex = showImage;
}
void ScanningWorkerThread::InitUIPointer(QSlider* scanNumSlider, QPushButton* button, QLineEdit* lineEdt)
{
	ScanNumSlider = scanNumSlider;
	ScanButton = button;
	SaveLocationText = lineEdt;
}

// 外部掃描的 Function
void ScanningWorkerThread::SetParams(QString* EndText, bool Save_Single_RawData, bool Save_Multi_RawData, bool Save_ImageData, bool Delete_ShakeData)
{
	EndScanText = EndText;
	NeedSave_Multi_RawData = Save_Single_RawData;
	NeedSave_Multi_RawData = Save_Multi_RawData;
 	NeedSave_ImageData = Save_ImageData;
	AutoDelete_ShakeData = Delete_ShakeData;
}
void ScanningWorkerThread::SetScanModel(bool IsStart)
{
	if (IsStart && ScanThread == nullptr)
	{
		// 開始掃描模式
		IsEnd = false;

		// 清一次記憶體
		System::GC::Collect();
		Last_PointType_1D->IsEmpty = true;

		ThreadStart^ threadDelegaate = gcnew ThreadStart(this, &ScanningWorkerThread::ScanProcess);
		ScanThread = gcnew Thread(threadDelegaate);
		ScanThread->Start();
	}
	else if(!IsStart)
	{
		// 結束掃描模式
		IsEnd = true;
	}
}

// 掃描相關
void ScanningWorkerThread::ScanProcess()
{
	// 掃描
	// 這邊要改進讀條 & 顯示文字
	bool ShowSingleScanDetail = false;
	bool ShowMultiScanDetail = false;
	#pragma region 關閉 Result 的 Define
	#ifndef DISABLE_SINGLE_RESULT
	ShowSingleScanDetail = true;
	#endif
	#ifndef DISABLE_MULTI_RESULT
	ShowMultiScanDetail = true;
	#endif
	#pragma endregion
	#pragma region 掃描的九軸的檔案
	GyroFile = new QFile(QDir(SaveLocationText->text()).absoluteFilePath("Gyro.txt"));
	GyroFile->open(QIODevice::WriteOnly);

	QTextStream ss(GyroFile);
	#pragma endregion
	while (!IsEnd)
	{
		#pragma region 1. 開始掃描的初始化設定
		QString SaveLocation;							// 最後儲存的路徑

		QTime currentTime = QTime::currentTime();
		QString TimeFileName = currentTime.toString("hh_mm_ss_zzz");
		SaveLocation = QDir(SaveLocationText->text()).absoluteFilePath(TimeFileName);
		//cout << "儲存位置: " << SaveLocation.toStdString() << endl;
		#pragma endregion
		#pragma region 2. 掃描單張資料
		(*ScanSingleDataFromDeviceV2)(SaveLocation + "_single", NeedSave_Single_RawData);
		(*TransformToIMG)(NeedSave_ImageData);
		#pragma endregion
		#pragma region 3. 顯示
		// 顯示畫面
		(*ShowImageIndex)();
		ScanNumSlider->setEnabled(false);
		#pragma endregion
		#pragma region 4. 如果是一張，那還要再掃一張，如果大於一張，驗證最後兩張是否正確
		// 複製下來給下一次做判斷
		if (Last_PointType_1D->IsEmpty)
		{
			// 如果只有一張
			Last_PointType_1D->IsEmpty = false;
			(*CopySingleBorder)(Last_PointType_1D->TypeData);
			continue;
		}
		else
		{
			// 有兩張以上，可以做比較
			bool IsShake = (*ShakeDetect_Single)(Last_PointType_1D->TypeData, ShowSingleScanDetail);
			if (IsShake)
			{
				// 如果有晃動，就要重新更新
				(*CopySingleBorder)(Last_PointType_1D->TypeData);
				continue;
			}
		}
		#pragma endregion
		#pragma region 5. 開始掃描多個資料
		(*ScanMultiDataFromDeviceV2)(SaveLocation + "_Multi", NeedSave_Multi_RawData);
		(*TransformToIMG)(NeedSave_ImageData);

		// 拿旋轉矩陣
		QQuaternion currentQuat = (*GetQuaternionFromDevice)();
		#pragma endregion
		#pragma region 6. 顯示
		(*ShowImageIndex)();
		ScanNumSlider->setEnabled(true);
		#pragma endregion
		#pragma region 7. 驗證是否有晃動到
		// 這裡只需要做一整片的驗證
		bool IsShake = (*ShakeDetect_Multi)(ShowMultiScanDetail);

		// 晃到重掃
		if (IsShake)
		{
			if (NeedSave_Multi_RawData && AutoDelete_ShakeData)
			{
				QFile file(SaveLocation + "_Multi");
				cout << "刪除檔案 " << (SaveLocation + "_Multi").toStdString() << endl;
				file.remove();
			}
			continue;
		}
		#pragma endregion
		#pragma region 8. 如果沒有晃到，那就儲存點雲 & 如果大於二就執行拼接
		cout << "可用資料!!" << endl;

		// 寫出九軸資訊
		ss << TimeFileName << " " << currentQuat.scalar() << " " << currentQuat.x() << " " << currentQuat.y() << " " << currentQuat.z() << endl;

		(*SavePointCloud)(currentQuat);
		(*AlignmentPointCloud)();
		#pragma endregion
	}

	#pragma region 結束按鈕設定
	ScanButton->setText(*EndScanText);
	ScanThread = nullptr;
	#pragma endregion
	#pragma region 清除資料
	// 關閉資料
	GyroFile->close();

	delete GyroFile;
	#pragma endregion
}
