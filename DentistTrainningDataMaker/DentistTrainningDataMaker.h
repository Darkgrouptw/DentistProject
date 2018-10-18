#pragma once
#include <iostream>
#include <QtWidgets/QMainWindow>
#include <QFileDialog>

#include "ui_DentistTrainningDataMaker.h"

#include "RawDataManager.h"

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

	// OCT
	void ReadRawDataToImage();
};
