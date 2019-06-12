#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_DentistDNNDemo.h"

#include <iostream>

#include <QVector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class DentistDNNDemo : public QMainWindow
{
	Q_OBJECT

public:
	DentistDNNDemo(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistDNNDemoClass ui;

private slots:
	//////////////////////////////////////////////////////////////////////////
	// 測試相關 Function
	//////////////////////////////////////////////////////////////////////////
	void TestRenderFunctionEvent();
};
