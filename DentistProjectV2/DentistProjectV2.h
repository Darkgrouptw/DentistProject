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
	QVector<QString> ExceptCOMName = { "COM1", "COM6" };
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
	// 藍芽、九軸測試
	//////////////////////////////////////////////////////////////////////////
	void PointCloudAlignmentTest();												// 測試點雲能不能拼接起來

	//////////////////////////////////////////////////////////////////////////
	// OCT 相關(主要)
	//////////////////////////////////////////////////////////////////////////
	void ChooseSaveLocaton();
	void AutoSaveWhileScan_ChangeEvent(int);									// UI 是否勾選(掃描自動儲存 Raw Data & Image 的通知)
	void ScanOCTMode();															// 掃描按鈕
	void ScanOCTOnceMode();														// 只掃描一張

	//////////////////////////////////////////////////////////////////////////
	// OCT 測試
	//////////////////////////////////////////////////////////////////////////
	void ReadRawDataToImage();													// 轉圖 & 儲存
	void ReadRawDataForBorderTest();											// 邊界測試 & 不儲存
	void ReadSingleRawDataForShakeTest();										// 偵測是否有晃動
	void ReadMultiRawDataForShakeTest();										// 偵測是否有晃動
	void SlimLabviewRawData();													// 縮小 Labview 掃出來的 Data

	//////////////////////////////////////////////////////////////////////////
	// 點雲操作相關
	//////////////////////////////////////////////////////////////////////////
	void PCIndexChangeEvnet(int);												// 更換 PC Index 的 Function
	void ReadPCEvent();															// 讀取 Point Cloud
	void SavePCEvent();															// 儲存 Point Cloud
	void DeletePCEvent();														// 刪除 Point Cloud
	void AlignLastTwoPCEvent();													// 拚前兩塊
	void CombineLastTwoPCEvent();												// 合併後面兩片
	void CombineAllPCEvent();													// 合併全部並輸出點雲
	void AlignmentAllPCTestEvent();												// 旋轉拼接測試
	void TransformMultiDataToPCEvent();											// 轉掃描資料

	//////////////////////////////////////////////////////////////////////////
	// Network 相關
	//////////////////////////////////////////////////////////////////////////
	void NetworkDataGenerateV2();												// 產生類神經網路
	void PredictResultTesting();												// 測試預測的結果

	//////////////////////////////////////////////////////////////////////////
	// Volumne Render 的測試
	//////////////////////////////////////////////////////////////////////////
	void VolumeRenderTest();													// 測試畫出來的部分

	//////////////////////////////////////////////////////////////////////////
	// 顯示部分的事件
	//////////////////////////////////////////////////////////////////////////
	void ScanNumSlider_Change(int);												// 這個是右邊視窗的顯示
	void DisplayPanelUpdate();													// 這個是 GL 視窗的更新
	void OCTViewOptionChange(int);												// 看的方向改變
};
