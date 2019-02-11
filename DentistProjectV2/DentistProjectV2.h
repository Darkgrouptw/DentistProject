#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_DentistProjectV2.h"

#include <iostream>
#include <QFileDialog>
#include <QTimer>
#include <QFile>
#include <QDir>
#include <QDate>
#include <QTime>
#include <QVector>
#include <QMessageBox>

#include "GlobalDefine.h"
#include "RawDataManager.h"

using namespace std;

class DentistProjectV2 : public QMainWindow
{
	Q_OBJECT

public:
	DentistProjectV2(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistProjectV2Class ui;

	//////////////////////////////////////////////////////////////////////////
	// 掃描相關變數
	//////////////////////////////////////////////////////////////////////////
	QString StartScanText;
	QString EndScanText;

	//////////////////////////////////////////////////////////////////////////
	// 其他變數 or 元件
	//////////////////////////////////////////////////////////////////////////
	RawDataManager	rawManager;													// 所有跟裝置有關的 (藍芽、OCT)
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");
	QTimer*	UpdateGLTimer = NULL;

	//////////////////////////////////////////////////////////////////////////
	// 藍芽相關參數
	//////////////////////////////////////////////////////////////////////////
	QString	RotationModeON_String = "Rotation Mode (ON)";
	QString RotationModeOFF_String = "Rotation Mode (OFF)";
	QString BLEDeviceName = "PenBLE";
	QString BLEDeviceAddress = "88:C2:55:9D:05:67";
	#ifdef TEST_NO_OCT
	QVector<QString> ExceptCOMName = { "COM1" };
	#else
	// 醫院那邊要例外排除的
	QVector<QString> ExceptCOMName = { "COM1" };
	#endif

private slots:
	//////////////////////////////////////////////////////////////////////////
	// 藍芽
	//////////////////////////////////////////////////////////////////////////
	void SearchCOM();															// 找 COM
	void ConnectCOM();															// 連結 COM
	void ScanBLEDevice();														// 找可用的 BLE Device
	void ConnectBLEDeivce();													// 連結
	void SetRotationMode();														// Rotation 設定模式
	void GyroResetToZero();														// 九軸歸零
	void ConnectBLEDevice_OneBtn();												// 一鍵建立連結

	//////////////////////////////////////////////////////////////////////////
	// OCT 相關(主要)
	//////////////////////////////////////////////////////////////////////////
	void ChooseSaveLocaton();
	void AutoSaveWhileScan_ChangeEvent(int);									// UI 是否勾選(掃描自動儲存 Raw Data & Image 的通知)
	void ScanOCTMode();															// 掃描按鈕

	//////////////////////////////////////////////////////////////////////////
	// OCT 測試
	//////////////////////////////////////////////////////////////////////////
	void ReadRawDataToImage();													// 轉圖 & 儲存
	void ReadRawDataForBorderTest();											// 邊界測試 & 不儲存
	void ReadSingleRawDataForShakeTest();										// 偵測是否有晃動
	void ReadMultiRawDataForShakeTest();										// 偵測是否有晃動

	//////////////////////////////////////////////////////////////////////////
	// 顯示部分的事件
	//////////////////////////////////////////////////////////////////////////
	void ScanNumSlider_Change(int);												// 這個是右邊視窗的顯示
	void DisplayPanelUpdate();													// 這個是 GL 視窗的更新
};
