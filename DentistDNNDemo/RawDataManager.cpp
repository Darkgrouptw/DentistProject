#include <QOpenGLWidget>				// 因為會跟 OpenCV 3 衝突
#include "RawDataManager.h"

RawDataManager::RawDataManager()
{

}
RawDataManager::~RawDataManager()
{

}

// OCT 相關的步驟
RawDataType RawDataManager::ReadRawDataFromFileV2(QString FileName)
{
	// 起始
	clock_t startT, endT;
	startT = clock();

	QFile inputFile(FileName);
	if (!inputFile.open(QIODevice::ReadOnly))
	{
		cout << "Raw Data 讀取錯誤" << endl;
		return RawDataType::ERROR_TYPE;
	}
	else
		cout << "讀取 Raw Data: " << FileName.toLocal8Bit().toStdString() << endl;

	int bufferSize = inputFile.size() / sizeof(quint8);

	QDataStream qData(&inputFile);
	QByteArray buffer;
	buffer.clear();
	buffer.resize(bufferSize);
	qData.readRawData(buffer.data(), bufferSize);

	inputFile.close();

	// 結算
	endT = clock();
	cout << "讀取時間: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << " sec" << endl;

	RawDataProperty prop = DManager.prop;
	if (bufferSize <= prop.SizeX * prop.SizeZ * 8)
	{
		//RawDataManager
		cout << "讀取單張 RawData !!" << endl;
		cudaV2.SingleRawDataToPointCloud(
			buffer.data(), bufferSize,
			prop.SizeX, prop.SizeZ,
			prop.ShiftValue, prop.K_Step, prop.CutValue);
		return RawDataType::SINGLE_DATA_TYPE;
	}
	else
	{
		//RawDataManager
		cout << "讀取多張 RawData !!" << endl;
		cudaV2.MultiRawDataToPointCloud(
			buffer.data(), bufferSize,
			prop.SizeX, prop.SizeY, prop.SizeZ,
			prop.ShiftValue, prop.K_Step, prop.CutValue);
		return RawDataType::MULTI_DATA_TYPE;
	}
}
void RawDataManager::TransformToIMG(bool NeedSave_Image = false)
{
#pragma region 開始時間
#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	clock_t startT, endT;
	startT = clock();
#endif
#pragma endregion
#pragma region 清空其他 Array
	// 如果跑出結果是全黑的，那有可能是顯卡記憶體不夠的問題
	ImageResultArray.clear();
	BorderDetectionResultArray.clear();

	QImageResultArray.clear();
	QBorderDetectionResultArray.clear();

	NetworkResultArray.clear();
	QNetworkResultArray.clear();
#pragma endregion
#pragma region 塞進 Array
	vector<Mat> TempMatArray;

	// 轉換到 Vector 中
	TempMatArray = cudaV2.TransfromMatArray(false);
	ImageResultArray = QVector<Mat>::fromStdVector(TempMatArray);

	TempMatArray = cudaV2.TransfromMatArray(true);
	BorderDetectionResultArray = QVector<Mat>::fromStdVector(TempMatArray);

	// 轉 QImage
	for (int i = 0; i < ImageResultArray.size(); i++)
	{
		QImage tempQImage = Mat2QImage(ImageResultArray[i], CV_8UC3);
		QImageResultArray.push_back(tempQImage);

		tempQImage = Mat2QImage(BorderDetectionResultArray[i], CV_8UC3);
		QBorderDetectionResultArray.push_back(tempQImage);

		if (NeedSave_Image)
		{
			// 原圖
			cv::imwrite("Images/OCTImages/origin_v2/" + to_string(i) + ".png", ImageResultArray[i]);

			// Combine 結果圖
			cv::imwrite("Images/OCTImages/border_v2/" + to_string(i) + ".png", BorderDetectionResultArray[i]);
		}
	}
#pragma endregion
#pragma region 結束時間
#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	endT = clock();

	if (NeedSave_Image)
		cout << "有存出圖片";
	else
		cout << "無存出圖片";
	cout << "，轉圖檔完成: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << "s" << endl;
#endif
#pragma endregion
}
void RawDataManager::TransformToOtherSideView()
{
	assert(ImageResultArray.size() > 1 && "呼叫此函式必須要有多張圖才可以呼叫");
	#pragma region 開始時間
	#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	clock_t startT, endT;
	startT = clock();
	#endif
	#pragma endregion
	#pragma region TopView
	Mat result = cudaV2.TransformToOtherSideView();
	//QImage qresult = Mat2QImage(result, CV_8UC3);
	//OtherSideResult->setPixmap(QPixmap::fromImage(qresult));

	// 順便存一分到 外圍
	cvtColor(result, OtherSideMat, CV_BGR2GRAY);
	#pragma endregion
	#pragma region 結束時間
	#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	endT = clock();
	cout << "TopView 轉換時間: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << "s" << endl;
	#endif
	#pragma endregion
}

// Network or Volume 相關的 Function
void RawDataManager::NetworkDataGenerateInRamV2()
{
	#pragma region 例外判斷
	if (ImageResultArray.size() != 250)
	{
		cout << "資料必須是立體資料!!" << endl;
		return;
	}
	#pragma endregion
	#pragma region 產生 Bounding Box
	// 更新點雲
	TLPointArray.clear();
	BRPointArray.clear();

	// 抓取 Bounding Box
	QFile BoundingBoxFile("./Predicts/boundingBox.txt");
	BoundingBoxFile.open(QIODevice::WriteOnly);
	QTextStream ss(&BoundingBoxFile);

	ss << "TopLeft (x, y), ButtomRight (x, y)" << endl;
	// 整個 Bounding Box Parse 過去
	for (int i = 0; i < ImageResultArray.size(); i++)
	{
		QVector2D topLeft, buttomRight;
		GetBoundingBox(ImageResultArray[i], topLeft, buttomRight);
		TLPointArray.push_back(topLeft);
		BRPointArray.push_back(buttomRight);

		ss << topLeft.x() << " " << topLeft.y() << " " << buttomRight.x() << " " << buttomRight.y() << endl;
	}
	BoundingBoxFile.close();
	#pragma endregion
}
bool RawDataManager::CheckIsValidData()
{
	if (TLPointArray.size() != 250 || BRPointArray.size() != 250)
		return false;

	QVector2D BoundingBox = BRPointArray[60] - TLPointArray[60];
	if (BoundingBox.x() < 150 || BoundingBox.y() < 150)
		return false;
	return true;
}
void RawDataManager::SaveNetworkImage()
{
	QDir NetworkUpLoadPath = "./Predicts/";

	QString tempImgPath = NetworkUpLoadPath.filePath("OtherSide.png");
	cv::imwrite(tempImgPath.toLocal8Bit().toStdString(), OtherSideMat);

	for (int i = 0; i <= 249; i++)
	{
		QVector2D TL = TLPointArray[i];
		QVector2D BR = BRPointArray[i];
		int width = BR[0] - TL[0];
		int height = BR[1] - TL[1];
		cv::imwrite(NetworkUpLoadPath.filePath(QString::number(i) + ".png").toLocal8Bit().toStdString(),
			cv::Mat(ImageResultArray[i], cv::Rect(TL[0], TL[1], width, height)));
	}
}
void RawDataManager::LoadPredictImage()
{
	QString testPath = "./Predicts";

		for (int i = 0; i <= 249; i++)
		{
			cv::Mat BlankImg = cv::Mat::zeros(ImageResultArray[0].size(), CV_8UC3);
			cv::Mat LoadImage = cv::imread((testPath + "/Result_" + QString::number(i) + ".png").toLocal8Bit().toStdString(), CV_LOAD_IMAGE_COLOR);

			QVector2D TL = TLPointArray[i];
			QVector2D BR = BRPointArray[i];
			int width = BR[0] - TL[0];
			int height = BR[1] - TL[1];

			LoadImage.copyTo(BlankImg(cv::Rect(TL[0], TL[1], width, height)));
			NetworkResultArray.push_back(BlankImg);
		}
}
void RawDataManager::SmoothNetworkData()
{
	clock_t smoothtime = clock();

#pragma region 找出最大最小值
	// r => Image rows
	// y => 張數
	// c => Image cols
	// 0, 0 => 是圖片的左上角
	int rMin = INT_MAX, rMax = 0,
		yMin = 0, yMax = 250,
		cMin = INT_MAX, cMax = 0;

	//assert(NetworkResultArray.size() == (200 - 60 + 1) && "必須要有 141 張圖!!");
	assert(NetworkResultArray.size() == (250 - 0 + 1) && "必須要有 251 張圖!!");

	for (int i = 0; i < NetworkResultArray.size(); i++)
	{
		int index = i;					// Offset 60 張圖

											// 取出點
		QVector2D TL = TLPointArray[i];
		QVector2D BR = BRPointArray[i];

		if (cMin > TL.x()) cMin = TL.x();
		if (rMin > TL.y()) rMin = TL.y();
		if (cMax < BR.x()) cMax = BR.x();
		if (rMax < BR.y()) rMax = BR.y();
	}

	// Clamp 到結果之間
	cMax = clamp(cMax, 0, DManager.prop.SizeZ / 2 - 1);
	rMax = clamp(rMax, 0, DManager.prop.SizeX - 1);

	cout << "Row: " << rMin << " " << rMax << " Col Max: " << cMin << " " << cMax << endl;
#pragma endregion
#pragma region Cuda 的部分
	std::vector<cv::Mat> NetworkResultSmooth = NetworkResultArray.toStdVector();

	// NetworkResultSmooth type為16 => CV_8UC3
	utilityTools.SetImageData(NetworkResultSmooth, cMin, rMin, cMax, rMax);

	std::vector<cv::Mat> TestResult = utilityTools.TransfromMatArray();

	smoothtime = clock() - smoothtime;

	cout << "Smooth花費時間 : " << ((float)smoothtime) / CLOCKS_PER_SEC << " sec" << endl;

	NetworkResultArray.clear();

	int width = cMax - cMin + 1;	// cols
	int height = rMax - rMin + 1;	// rows

	for (int i = 0; i < TestResult.size(); i++)
	{
		// 寫出圖片
		cv::imwrite("./Predicts/Smooth/" + to_string(i) + ".png", TestResult[i]);
		cv::Mat BlankImg = cv::Mat::zeros(ImageResultArray[0].size(), CV_8UC3);

		TestResult[i].copyTo(BlankImg(cv::Rect(cMin, rMin, width, height)));
		NetworkResultArray.push_back(BlankImg);
	}
#pragma endregion
}
void RawDataManager::NetworkDataToQImage()
{
	for (int i = 0; i < NetworkResultArray.size(); i++)
	{
		QImage qimg = Mat2QImage(NetworkResultArray[i], CV_8UC3);
		QNetworkResultArray.append(qimg);
	}
}

// 網路
Mat RawDataManager::GetBoundingBox(Mat img, QVector2D& TopLeft, QVector2D& ButtomRight)
{
	#pragma region 抓出資料，並做模糊
	Mat imgGray;
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	blur(imgGray, imgGray, BlurSize);
	#pragma endregion
	#pragma region 根據閘值，去抓邊界
	// 先根據閘值，並抓取邊界
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	threshold(imgGray, threshold_output, BoundingThreshold, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// 先給占存 Array
	vector<BoundingBoxDataStructRaw> dataInfo(contours.size());
	#pragma endregion
	#pragma region 抓出最大的框框
	// 抓出擬合的結果
	for (size_t img = 0; img < contours.size(); img++)
	{
		BoundingBoxDataStructRaw data;
		approxPolyDP(Mat(contours[img]), data.contoursPoly, 3, true);
		data.boundingRect = boundingRect(Mat(data.contoursPoly));

		// 加進陣列
		dataInfo[img] = data;
	}
	Mat drawing = img.clone();
	std::sort(dataInfo.begin(), dataInfo.end(), SortByContourPointSize);

	// 抓出最亮，且最大的
	int i = 0;
	vector<vector<Point>> contoursPoly(1);
	contoursPoly[0] = dataInfo[i].contoursPoly;

	// 確定這邊可以抓到
	drawContours(drawing, contoursPoly, (int)i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());

	// 邊界
	Point tl = dataInfo[i].boundingRect.tl();
	tl.x = max(0, tl.x - BoundingOffset);
	tl.y = max(0, tl.y - BoundingOffset);
	Point br = dataInfo[i].boundingRect.br();
	br.x = min(DManager.prop.SizeZ, br.x + BoundingOffset);
	br.y = min(DManager.prop.SizeX, br.y + BoundingOffset);

	rectangle(drawing, tl, br, Scalar(0, 255, 255), 2, 8, 0);

	TopLeft.setX(tl.x);
	TopLeft.setY(tl.y);
	ButtomRight.setX(br.x);
	ButtomRight.setY(br.y);
	#pragma endregion
	return drawing;
}
bool RawDataManager::SortByContourPointSize(BoundingBoxDataStructRaw& c1, BoundingBoxDataStructRaw& c2)
{
	return c1.boundingRect.area() > c2.boundingRect.area();
}

// Helper Function
QImage RawDataManager::Mat2QImage(cv::Mat const& src, int Type)
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
int RawDataManager::clamp(int value, int min, int max)
{
	return std::max(min, std::min(value, max));
}