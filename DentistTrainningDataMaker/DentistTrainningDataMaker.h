#pragma once
#include <iostream>
#include <QtWidgets/QMainWindow>
#include <QFileDialog>
#include <QDir>
#include <QDate>
#include <QTime>

#include "ui_DentistTrainningDataMaker.h"

#include "GlobalDefine.h"
#include "RawDataManager.h"

#include <QMessageBox>
using namespace std;

class DentistTrainningDataMaker : public QMainWindow
{
	Q_OBJECT

public:
	DentistTrainningDataMaker(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistTrainningDataMakerClass ui;

	//QFileDialog dia
	RawDataManager rawManager;

	// 儲存時，如果不溝時間，會以 Index 儲存
	int ScanIndex = 0;
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");

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

	// OCT 相關
	void ReadRawDataToImage();
	void ReadRawDataForBorderTest();				// 邊界測試
	void ChooseSaveLocaton();
	void SaveWithTime_ChangeEvent(int);				// UI 是否勾選(儲存加上時間)
	void AutoSaveWhileScan_ChangeEvent(int);		// UI 是否勾選(掃描自動儲存)
	void ScanOCT();									// 掃描按鈕
	void BeepSoundTest();							// 測試掃描時會使用的 Beep Sound

	// 顯示部分的事件
	void ScanNumSlider_Change(int);
};
