#include "DentistDNNDemo.h"

DentistDNNDemo::DentistDNNDemo(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	#pragma region 事件連結
	connect(ui.TestRenderingBtn, SIGNAL(clicked()), this, SLOT(TestRenderFunctionEvent()));
	#pragma endregion
}

// 測試相關函式
void DentistDNNDemo::TestRenderFunctionEvent()
{
	#pragma region Test 路徑
	QString TestFilePath = "E:/DentistData/DentistProjectV2-p3dLon/";
	QString OtherSidePath_Predict = TestFilePath + "Predict.png";
	QString OtherSidePath_Org = TestFilePath + "OtherSide.png";
	#pragma endregion
	#pragma region 讀圖
	Mat otherSideMat_Org		= imread(OtherSidePath_Org.toLocal8Bit().toStdString(), IMREAD_GRAYSCALE);
	Mat otherSideMat_Predict	= imread(OtherSidePath_Predict.toLocal8Bit().toStdString(), IMREAD_GRAYSCALE);
	QVector<Mat> FullMat;

	for (int i = 0; i <= 140; i++)
	{
		Mat mat = imread((TestFilePath + QString::number(i) + ".png").toLocal8Bit().toStdString(), IMREAD_COLOR);
		FullMat.push_back(mat);
	}
	((OpenGLWidget*)(ui.DisplayPanel))->ProcessImg(otherSideMat_Org, otherSideMat_Predict);
	#pragma endregion
	#pragma region 刷新
	ui.DisplayPanel->update();
	#pragma endregion
}