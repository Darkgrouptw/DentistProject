#include "DentistDNNDemo.h"

DentistDNNDemo::DentistDNNDemo(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	#pragma region 事件連結
	// 主要功能
	connect(ui.slidingBar,			SIGNAL(valueChanged(int)),	this, SLOT(SliderValueChange(int)));

	// 測試相關
	connect(ui.TestRenderingBtn,	SIGNAL(clicked()),			this, SLOT(TestRenderFunctionEvent()));
	#pragma endregion
	#pragma region UI 初始化
	QImage img("./Images/ColorMap.png");
	ui.ColorMapLabel->setPixmap(QPixmap::fromImage(img));
	#pragma endregion
}

// 主要功能
void DentistDNNDemo::SliderValueChange(int)
{
	ui.DisplayPanel->GetSliderValue(ui.slidingBar->value());
	ui.ColorMapCurrentValue->setText(ui.DisplayPanel->GetColorMapValue(ui.slidingBar->value()));
	ui.DisplayPanel->update();
}

// 測試相關函式
void DentistDNNDemo::TestRenderFunctionEvent()
{
	#pragma region Test 路徑
	QString TestFilePath = "E:/DentistData/DentistProjectV2-p3dLon/";
	QString OtherSidePath_Predict = TestFilePath + "Predict.png";
	QString OtherSidePath_Org = TestFilePath + "OtherSide.png";
	QString BoundingBoxPath = TestFilePath + "boundingBox.txt";
	#pragma endregion
	#pragma region 讀 Bounding Box
	ReadBounding(BoundingBoxPath);
	#pragma endregion	
	#pragma region 讀圖
	Mat otherSideMat_Org		= imread(OtherSidePath_Org.toLocal8Bit().toStdString(), IMREAD_GRAYSCALE);
	Mat otherSideMat_Predict	= imread(OtherSidePath_Predict.toLocal8Bit().toStdString(), IMREAD_GRAYSCALE);
	QVector<Mat> FullMat;

	for (int i = 0; i <= 140; i++)
	{
		Mat mat = imread((TestFilePath + "Smooth/" + QString::number(i) + ".png").toLocal8Bit().toStdString(), IMREAD_COLOR);
		FullMat.push_back(mat);
	}
	((OpenGLWidget*)(ui.DisplayPanel))->ProcessImg(otherSideMat_Org, otherSideMat_Predict, FullMat, OrginTL, OrginBR, ui.ColorMapMaxValue, ui.ColorMapMinValue);
	#pragma endregion
	#pragma region 刷新
	ui.DisplayPanel->GetSliderValue(ui.slidingBar->value());
	ui.ColorMapCurrentValue->setText(ui.DisplayPanel->GetColorMapValue(ui.slidingBar->value()));
	ui.DisplayPanel->update();
	#pragma endregion
}

void DentistDNNDemo::ReadBounding(QString FileName) {
	#pragma region 讀取BoundingBox
	QFile file(FileName);
	file.open(QIODevice::ReadOnly);
	cout << "讀取boundingbox: " << FileName.toLocal8Bit().toStdString() << endl;

	// 初始化變數
	float a, b, c;
	QTextStream ss(&file);

	QVector<QVector2D> TL;
	QVector<QVector2D> BR;

	QString TempFile;

	float w, x, y, z;
	// 第一行不要
	TempFile = ss.readLine();

	while (!ss.atEnd())
	{
		// 讀一條
		TempFile = ss.readLine();
		if (TempFile == "")
			break;

		//xyz
		assert(TempStr.size == 3 && "讀取的資料有誤!!");
		w = TempFile.section(' ', 0, 0).trimmed().toFloat();
		x = TempFile.section(' ', 1, 1).trimmed().toFloat();
		y = TempFile.section(' ', 2, 2).trimmed().toFloat();
		z = TempFile.section(' ', 3, 3).trimmed().toFloat();

		//cout << w << " " << x << " " << y << " " << z << endl;

		TL.push_back(QVector2D(w, x));
		BR.push_back(QVector2D(y, z));
	}

	// 關閉檔案
	file.close();

	cout << "讀取BoundingBox完成!!" << endl;
	#pragma endregion
	#pragma region 得到最大boundbox range
	for (int i = 60 - 1; i <= 200 - 1; i++)
	{
		if (TL[i].x() < OrginTL.x())OrginTL.setX(TL[i].x());
		if (TL[i].y() < OrginTL.y())OrginTL.setY(TL[i].y());
		if (BR[i].x() > OrginBR.x())OrginBR.setX(BR[i].x());
		if (BR[i].y() > OrginBR.y())OrginBR.setY(BR[i].y());
	}
	#pragma endregion
}