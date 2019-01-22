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

	// 其他元件
	RawDataManager	rawManager;								// 所有跟裝置有關的 (藍芽、OCT)
};
