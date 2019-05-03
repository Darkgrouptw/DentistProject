#pragma once
#include "TensorflowNet.h"
#include "Display_TopView.h"
#include "Display_BoundingBox.h"

#include <QDir>
#include <QFileDialog>
#include <QTextCodec>

#include <QtWidgets/QMainWindow>
#include "ui_DentistLabelTool.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

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

	
	QVector<QImage> QBoundImgVec;		//boundbox image

private slots:
	//////////////////////////////////////////////////////////////////////////
	// 檔案相關
	//////////////////////////////////////////////////////////////////////////
	void ReadBoundingBox();
	QImage Mat2QImage(cv::Mat const& src, int Type);

	//////////////////////////////////////////////////////////////////////////
	// 顯示部分的事件
	//////////////////////////////////////////////////////////////////////////
	void ScanNumSlider_Change(int);

	//void Imagelayerchange();
};
