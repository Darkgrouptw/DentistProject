#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_DentistDNNDemo.h"

#include <iostream>

#include <QVector>
#include <QVector2D>
#include <QFile>

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

	// Read BoundingBoxFunc
	void ReadBounding(QString);
	// Bounding
	QVector2D OrginTL = QVector2D(9999, 9999);
	QVector2D OrginBR = QVector2D(-1, -1);

private slots:
	//////////////////////////////////////////////////////////////////////////
	// 測試相關 Function
	//////////////////////////////////////////////////////////////////////////
	void TestRenderFunctionEvent();
};
