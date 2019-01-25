﻿#pragma once
/*
這邊是管理 System::Thread 的 class
由於這邊是使用 clr (Manage 的物件)
所以class 要用 public ref class (因為 clr 不支援 c++ thread & QThread)
*/
#include <iostream>
#include <vector>
#include <functional>

#include <QDir>
#include <QTime>
#include <QLineEdit>
#include <QString>
#include <QPushButton>

using namespace std;
using namespace System::Threading;

public ref class ScanningWorkerThread
{
public:
	ScanningWorkerThread();
	~ScanningWorkerThread();

	//////////////////////////////////////////////////////////////////////////
	// 傳送 Function Pointer
	//////////////////////////////////////////////////////////////////////////
	void InitFunctionPointer(function<void(QString, bool)>*,
		function<void(QString, bool)>*,
		function<void(bool)>*);													// 初始化 Function Pointer
	void InitUIPointer(QPushButton*, QLineEdit*);								// 初始化 UI Pointer (Thread 中，做完會去改圖，所以需要這個 Function Pointer)

	//////////////////////////////////////////////////////////////////////////
	// 外部掃描的 Function
	//////////////////////////////////////////////////////////////////////////
	void SetParams(QString*, bool, bool);										// 設定參數
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
	// 8. 傳到 UI 告訴他要顯示
	//////////////////////////////////////////////////////////////////////////
	void ScanProcess();															// Thread 跑的 Function
	bool IsEnd = true;															// 是否要結束
	bool NeedSave_RawData = false;												// 是否要儲存 Raw Data
	bool NeedSave_ImageData = false;											// 是否要儲存完 Image Data
	//const int Start


	//////////////////////////////////////////////////////////////////////////
	// UI 設定
	//////////////////////////////////////////////////////////////////////////
	QLineEdit* SaveLocationText;
	QString* EndScanText;
	QPushButton* ScanButton;

	//////////////////////////////////////////////////////////////////////////
	// 實際在執行的 Thread
	//////////////////////////////////////////////////////////////////////////
	Thread^ ScanThread;

	//////////////////////////////////////////////////////////////////////////
	// Function Pointer
	//////////////////////////////////////////////////////////////////////////
	function<void(QString, bool)>*	ScanSingleDataFromDeviceV2 = NULL;
	function<void(QString, bool)>*	ScanMultiDataFromDeviceV2 = NULL;
	function<void(bool)>*			TranformToIMG = NULL;
};

