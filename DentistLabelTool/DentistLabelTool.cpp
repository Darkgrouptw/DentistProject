#include "DentistLabelTool.h"

DentistLabelTool::DentistLabelTool(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	#pragma region 事件連接
	connect(ui.actionOpen,			SIGNAL(triggered()),	this,	SLOT(ReadBoundingBox()));
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
	
	double* _data = new double[img.rows * img.cols];
	double** data = new double*[img.rows];
	for (int i =0; i < img.rows; i++)
	{
		data[i] = &_data[i * img.cols];

		#pragma omp parallel for
		for (int j = 0; j < img.cols; j++)
		{
			int pixel = (int)img.at<uchar>(i, j);
			data[i][j] = pixel;
		}
	}

	//// 預測
	//double** PreidctData = net.Predict(data);
	//
	//// Uchar
	//uchar* ImgData = new uchar[img.rows * img.cols];
	//#pragma omp parallel for
	//for (int i =0; i < img.rows * img.cols; i++)
	//{
	//	int rowIndex = i / img.cols;
	//	int colIndex = i % img.cols;
	//	ImgData[i] = (uchar)(data[rowIndex][colIndex] * 255);
	//}
	//QImage qImage(ImgData, 250, 250, QImage::Format_Indexed8);
	//qImage.save("D:/a.png");

	delete[] _data;
	delete[] data;
	//delete[] ImgData;
	#pragma endregion
	#pragma region 預測 & 測試
	//PreidctData
	#pragma endregion
}

// Helper Function
//QImage DentistLabelTool::Mat2QImage(cv::Mat const& src, int Type)
//{
//	cv::Mat temp;												// make the same cv::Mat
//	if (Type == CV_8UC3)
//		cvtColor(src, temp, CV_BGR2RGB);						// cvtColor Makes a copt, that what i need
//	else if (Type == CV_32F)
//	{
//		src.convertTo(temp, CV_8U, 255);
//		cvtColor(temp, temp, CV_GRAY2RGB);						// cvtColor Makes a copt, that what i need
//	}
//	QImage dest((const uchar *)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
//	dest.bits();												// enforce deep copy, see documentation 
//																// of QImage::QImage ( const uchar * data, int width, int height, Format format )
//	return dest;
//}