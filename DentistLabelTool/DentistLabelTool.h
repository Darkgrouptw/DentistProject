#pragma once
#include "TensorflowNet.h"

#include <QDir>
#include <QFileDialog>
#include <QTextCodec>

#include <QtWidgets/QMainWindow>
#include "ui_DentistLabelTool.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//using namespace cv;

class DentistLabelTool : public QMainWindow
{
	Q_OBJECT

public:
	DentistLabelTool(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistLabelToolClass ui;

	//////////////////////////////////////////////////////////////////////////
	// Label 檔案相關
	//////////////////////////////////////////////////////////////////////////
	QImage img;
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");
	TensorflowNet net;

private slots:
	//////////////////////////////////////////////////////////////////////////
	// 檔案相關
	//////////////////////////////////////////////////////////////////////////
	void ReadBoundingBox();
};
