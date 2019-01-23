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

	// 儲存時，如果不溝時間，會以 Index 儲存
	int ScanIndex = 0;
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");

	// 其他元件
	RawDataManager	rawManager;								// 所有跟裝置有關的 (藍芽、OCT)

private slots:
	
	//////////////////////////////////////////////////////////////////////////
	// OCT 相關(主要)
	//////////////////////////////////////////////////////////////////////////
	void ChooseSaveLocaton();
	void SaveWithTime_ChangeEvent(int);						// UI 是否勾選(儲存加上時間)
	void AutoSaveWhileScan_ChangeEvent(int);				// UI 是否勾選(掃描自動儲存 Raw Data & Image 的通知)

	//////////////////////////////////////////////////////////////////////////
	// OCT 測試
	//////////////////////////////////////////////////////////////////////////
	void ReadRawDataToImage();								// 轉圖 & 儲存
	void ReadRawDataForBorderTest();						// 邊界測試 & 不儲存
	//void ReadRawDataForShakeTest();						// 偵測是否有晃動

	//////////////////////////////////////////////////////////////////////////
	// 顯示部分的事件
	//////////////////////////////////////////////////////////////////////////
	void ScanNumSlider_Change(int);						// 這個是右邊視窗的顯示
};
