#pragma once
#include <iostream>
#include <QtWidgets/QMainWindow>
#include <QFileDialog>
#include <QFile>
#include <QDir>
#include <QDate>
#include <QTime>
#include <QMessageBox>

#include "ui_DentistProject.h"

#include "GlobalDefine.h"
#include "RawDataManager.h"
#include "SegNetModel.h"

using namespace std;

class DentistProject : public QMainWindow
{
	Q_OBJECT

public:
	DentistProject(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistProjectClass ui;

	// 其他元件
	RawDataManager	rawManager;								// 所有跟裝置有關的 (藍芽、OCT)
	SegNetModel		segNetModel;							// SegNet Model

	// 儲存時，如果不溝時間，會以 Index 儲存
	int ScanIndex = 0;
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");

	// 一次送多少
	int GPUBatchSize = 30;

private slots:
	void LoadSTL();
	void ScanData();

	// Render Options
	void SetRenderTriangleBool();
	void SetRenderBorderBool();
	void SetRenderPointCloudBool();

	// 藍芽
	void SearchCOM();
	void ConnectCOM();
	void ScanBLEDevice();
	void ConnectBLEDeivce();

	// OCT 相關(主要)
	void ChooseSaveLocaton();
	void SaveWithTime_ChangeEvent(int);					// UI 是否勾選(儲存加上時間)
	void AutoSaveWhileScan_ChangeEvent(int);			// UI 是否勾選(掃描自動儲存 Raw Data & Image 的通知)
	void ScanOCT();										// 掃描按鈕

	// OCT 測試
	void ReadRawDataToImage();							// 轉圖 & 儲存
	void WriteRawDataForTesting();						// 寫出檔案
	void ReadRawDataForBorderTest();					// 邊界測試 & 不儲存
	void ReadRawDataForShakeTest();						// 偵測是否有晃動
	void BeepSoundTest();								// 測試掃描時會使用的 Beep Sound
	void SegNetTest();									// 測試網路的結果

	// 顯示部分的事件
	void ScanNumSlider_Change(int);
};
