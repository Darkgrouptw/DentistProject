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
	function<void(bool)>* ToImage)
{
	ScanSingleDataFromDeviceV2 = ScanSingle;
	ScanMultiDataFromDeviceV2 = ScanMulti;
	TranformToIMG = ToImage;
}
void ScanningWorkerThread::InitShowFunctionPointer(function<void()>* showImage)
{
	ShowImageIndex = showImage;
}
void ScanningWorkerThread::InitUIPointer(QSlider* scanNumSlider, QPushButton* button, QLineEdit* lineEdt)
{
	ScanNumSlider = scanNumSlider;
	ScanButton = button;
	SaveLocationText = lineEdt;
}

// 外部掃描的 Function
void ScanningWorkerThread::SetParams(QString* EndText, bool Save_RawData, bool Save_ImageData)
{
	EndScanText = EndText;
	NeedSave_RawData = Save_RawData;
	NeedSave_ImageData = Save_ImageData;
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
	while (!IsEnd)
	{
		#pragma region 1. 開始掃描的初始化設定
		QString SaveLocation;							// 最後儲存的路徑

		QTime currentTime = QTime::currentTime();
		QString TimeFileName = currentTime.toString("hh_mm_ss_zzz");
		SaveLocation = QDir(SaveLocationText->text()).absoluteFilePath(TimeFileName);
		cout << "儲存位置: " << SaveLocation.toStdString() << endl;
		#pragma endregion
		#pragma region 2. 掃描單張資料
		(*ScanSingleDataFromDeviceV2)(SaveLocation + "_single", NeedSave_RawData);
		(*TranformToIMG)(NeedSave_ImageData);
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
			//Last_PointType_1D->IsEmpty = false;

			continue;
		}
		#pragma endregion
		#pragma region 5. 開始掃描多個資料
		//(*ScanMultiDataFromDeviceV2)(SaveLocation + "_Multi", NeedSave_RawData);
		//(*TranformToIMG)(NeedSave_ImageData);
		#pragma endregion
		#pragma region 6. 顯示
		//(*ShowImageIndex)();
		//ScanNumSlider->setEnabled(true);
		//break;
		#pragma endregion
		#pragma region 7. 驗證是否有晃動到
		#pragma endregion
		#pragma region 8. 傳到 UI 告訴他要顯示
		#pragma endregion
	}

	// 結束
	ScanButton->setText(*EndScanText);
	ScanThread = nullptr;
}
