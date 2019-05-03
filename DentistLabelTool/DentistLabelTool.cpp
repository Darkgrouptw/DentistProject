#include "DentistLabelTool.h"

DentistLabelTool::DentistLabelTool(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	#pragma region 事件連接
	connect(ui.actionOpen,			SIGNAL(triggered()),		this,	SLOT(ReadBoundingBox()));

	connect(ui.SliderPage,			SIGNAL(valueChanged(int)),	this,	SLOT(ScanNumSlider_Change(int)));

	#pragma endregion
}

// 檔案相關
void DentistLabelTool::ReadBoundingBox()
{
	#pragma region 讀取資料
	QString BoundingBoxLocation = QFileDialog::getOpenFileName(this, codec->toUnicode("讀取 boundingBox"), "E:/DentistData/NetworkData/", "boundingBox.txt", nullptr, QFileDialog::DontUseNativeDialog);

	int index = BoundingBoxLocation.lastIndexOf("/");
	BoundingBoxLocation = BoundingBoxLocation.left(index);
	#pragma endregion
	#pragma region Other Side
	QDir dir(BoundingBoxLocation);
	QString otherSideFile = dir.absoluteFilePath("OtherSide.png");

	cv::Mat img = cv::imread(otherSideFile.toStdString(), 0);
	cout << img.rows << " " << img.cols << endl;
	
	float* _data = new float[img.rows * img.cols];
	float** data = new float*[img.rows];
	for (int i =0; i < img.rows; i++)
	{
		data[i] = &_data[i * img.cols];

		#pragma omp parallel for
		for (int j = 0; j < img.cols; j++)
		{
			double pixel = (double)img.at<uchar>(i, j);
			data[i][j] = pixel;
		}
	}
	#pragma endregion
	#pragma region 預測 & 測試
	// 預測
	float** PreidctData = net.Predict(data);

	// Uchar
	uchar* ImgData = new uchar[img.rows * img.cols];
	int max = 0;
	#pragma omp parallel for
	for (int i = 0; i < img.rows * img.cols; i++)
	{
		int rowIndex = i / img.cols;
		int colIndex = i % img.cols;
		ImgData[i] = (uchar)(PreidctData[rowIndex][colIndex]);
		if (ImgData[i] > max)
			max = ImgData[i];
	}
	cout << "C++ Max: " << max << endl;

	// 設定成圖
	cv::Mat PredictImg(img.rows, img.cols, CV_8UC1, ImgData);
	cv::cvtColor(PredictImg.clone(), PredictImg, CV_GRAY2BGR);
	QImage QPredictImg = Mat2QImage(PredictImg, CV_8UC3);

	delete[] _data;
	delete[] data;
	delete[] ImgData;

	net.DeleteArray(PreidctData);
	#pragma endregion

	#pragma region Load BoundingBox Image

	////boundingbox texture
	QString boundboxpath = dir.absoluteFilePath("");

	vector<cv::Mat> BoundImgVec;

	for (int i = 60; i <= 200; i++) {
		BoundImgVec.push_back(cv::imread(boundboxpath.toStdString() + "/boundingBox_v2/" + std::to_string(i) + ".png", 0));
		cv::cvtColor(BoundImgVec[i - 60].clone(), BoundImgVec[i - 60], CV_GRAY2BGR);
		QBoundImgVec.push_back(Mat2QImage(BoundImgVec[i - 60], CV_8UC3));
	}

	#pragma endregion

	#pragma region 丟到 UI 上
	Display_TopView* widget1 = (Display_TopView*)ui.Widget1;
	widget1->LoadTexture(QPredictImg, 0);
	widget1->update();

	Display_BoundingBox* widget2 = (Display_BoundingBox*)ui.Widget2;
	widget2->LoadTexture(QBoundImgVec[0], 0);
	//widget2->SetFbo();
	//widget2->OpenPainter = true;
	widget2->update();
	#pragma endregion

}

// Helper Function
QImage DentistLabelTool::Mat2QImage(cv::Mat const& src, int Type)
{
	cv::Mat temp;												// make the same cv::Mat
	if (Type == CV_8UC3)
		cvtColor(src, temp, CV_BGR2RGB);						// cvtColor Makes a copt, that what i need
	else if (Type == CV_32F)
	{
		src.convertTo(temp, CV_8U, 255);
		cvtColor(temp, temp, CV_GRAY2RGB);						// cvtColor Makes a copt, that what i need
	}
	QImage dest((const uchar *)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
	dest.bits();												// enforce deep copy, see documentation 
																// of QImage::QImage ( const uchar * data, int width, int height, Format format )
	return dest;
}

// 顯示相關
void DentistLabelTool::ScanNumSlider_Change(int value)
{
	ui.Widget1->sliderValue = value;
	ui.Widget1->update();
	ui.Widget2->sliderValue = value;
	ui.Widget2->LoadTexture(QBoundImgVec[value -60], 0);
	ui.Widget2->update();
}