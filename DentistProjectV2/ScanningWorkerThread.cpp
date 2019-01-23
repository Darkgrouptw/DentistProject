#include "ScanningWorkerThread.h"

ScanningWorkerThread::ScanningWorkerThread()
{
}
ScanningWorkerThread::~ScanningWorkerThread()
{
}

// 傳送 Function Pointer
void ScanningWorkerThread::InitFunctionPointer(function<void(QString, bool)>* ScanSingle, function<void(QString, bool)>* ScanMulti)
{
	ScanSingleDataFromDeviceV2 = ScanSingle;
	ScanMultiDataFromDeviceV2 = ScanMulti;
}
void ScanningWorkerThread::InitUIPointer(QPushButton* button, QString* endText)
{
	ScanButton = button;
	EndScanText = endText;
}

// 外部掃描的 Function
void ScanningWorkerThread::SetScanModel(bool IsStart)
{
	if (IsStart && ScanThread == nullptr)
	{
		// 開始掃描模式
		IsEnd = false;

		// 清一次記憶體
		System::GC::Collect();

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
		#pragma region 1. 開始掃描，要顯示畫面
		//(*TestFunciton)(times++);
		//cout << times++ << endl;
		//Thread::Sleep(300);
		#pragma endregion
		#pragma region 2. 掃描單張資料

		#pragma endregion
		#pragma region 3. 等待結果判斷
		#pragma endregion
		#pragma region 4. 如果是一張，那還要再掃一張，如果大於一張，驗證最後兩張是否正確
		#pragma endregion
		#pragma region 5. 開始掃描多個資料
		#pragma endregion
		#pragma region 6. 等待結果判斷
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
