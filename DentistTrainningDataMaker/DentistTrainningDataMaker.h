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
	RawDataManager rawManager;

private slots:
	void LoadSTL();

	// Render Options
	void SetRenderTriangleBool();
	void SetRenderBorderBool();
	void SetRenderPointCloudBool();
};
