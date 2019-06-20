#include "OpenGLWidget.h"

OpenGLWidget::OpenGLWidget(QWidget* parent = 0) : QOpenGLWidget(parent)
{
}
OpenGLWidget::~OpenGLWidget()
{
}

// 繪畫相關函式
void OpenGLWidget::initializeGL()
{
	#pragma region 初始化參數
	initializeOpenGLFunctions();
	glClearColor(0.5f, 0.5, 0.5f, 1);

	// 連結 Shader
	Program = new QOpenGLShaderProgram();
	Program->addCacheableShaderFromSourceFile(QOpenGLShader::Vertex,	"./Shaders/Plane.vsh");
	Program->addCacheableShaderFromSourceFile(QOpenGLShader::Fragment,	"./Shaders/Plane.fsh");
	Program->link();
	#pragma endregion
	#pragma region Vertex & UV
	QVector<QVector3D> plane_vertices;
	plane_vertices.push_back(QVector3D(-1, -1, 0));
	plane_vertices.push_back(QVector3D(1, -1, 0));
	plane_vertices.push_back(QVector3D(1, 1, 0));
	plane_vertices.push_back(QVector3D(-1, 1, 0));

	QVector<QVector2D> plane_uvs;
	plane_uvs.push_back(QVector2D(0, 1));
	plane_uvs.push_back(QVector2D(1, 1));
	plane_uvs.push_back(QVector2D(1, 0));
	plane_uvs.push_back(QVector2D(0, 0));
	#pragma endregion
	#pragma region Bind Buffer
	// VertexBuffer
	glGenBuffers(1, &VertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, plane_vertices.size() * sizeof(QVector3D), plane_vertices.constData(), GL_STATIC_DRAW);

	// UV
	glGenBuffers(1, &UVBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, UVBuffer);
	glBufferData(GL_ARRAY_BUFFER, plane_uvs.size() * sizeof(QVector2D), plane_uvs.constData(), GL_STATIC_DRAW);
	#pragma endregion
}
void OpenGLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (OtherSideTexture != NULL)
	{
		Program->bind();

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, UVBuffer);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		Program->setUniformValue("texture", 0);
		Program->setUniformValue("probTexture", 1);
		Program->setUniformValue("colorMapTexture", 2);

		OtherSideTexture->bind(0);
		ProbTexture->bind(1);
		DepthTexture->bind(2);

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		OtherSideTexture->release();
		ProbTexture->release();
		DepthTexture->release();
		Program->release();
	}
}

// 外部呼叫函式
void OpenGLWidget::ProcessImg(Mat otherSide, Mat prob, QVector<Mat> FullMat, QVector2D OriginTL, QVector2D OriginBR, QLabel* MaxValueLabel, QLabel* MinValueLabel)
{
	#pragma region 反轉顏色
	threshold(prob.clone(), prob, 150, 255, THRESH_BINARY);

	bitwise_not(prob.clone(), prob);
	thin(prob, false, false, false);
	#pragma endregion
	#pragma region OtherSide Clone
	cvtColor(otherSide.clone(), otherSide, CV_GRAY2BGR);
	cvtColor(prob.clone(), prob, CV_GRAY2BGR);

	if (OtherSideTexture != NULL)
	{
		delete OtherSideTexture;
		delete ProbTexture;
	}
	#pragma endregion
	#pragma region 抓出結果
	// 每一張圖片的結果
	vector<Mat> ChannelMat;
	QVector2D TL, BR;
	QVector<int> LastY;
	//int LastY = -1;
	for (int i = 0; i < FullMat.size(); i++)
	{
		// 抓取藍色的部分，去抓取 Bounding Box
		split(FullMat[i], ChannelMat);
		Mat img = GetBoundingBox(ChannelMat[0], TL, BR);
		
		LastY.append(TL.y() + OriginTL.y());
	}
	#pragma endregion
	#pragma region 畫上結果
	// 畫結果
	for (int i = 1; i < LastY.size(); i++)
	{
		cv::Point LeftPoint = cv::Point(i - 1 + 60, LastY[i - 1]);
		cv::Point RightPoint = cv::Point(i + 60, LastY[i]);

		if (LastY[i - 1] != -1 && LastY[i] != -1)
			line(prob, LeftPoint, RightPoint, Scalar(0, 0, 0), 1);
	}
	#pragma endregion
	#pragma region 算出牙肉 & 齒槽骨的位置(Pixel)
	MeatBounding.clear();
	BoneBounding.clear();

	imwrite("D:/a.png", prob);
	QVector<int> nonZeroIndex;
	for (int col = 60; col <= 200; col++)
	{
		QVector<int> CanBeIndex;
		for (int row = 0; row < prob.rows; row++)
		{
			if (prob.at<Vec3b>(row, col)[0] == 0)
			{
				CanBeIndex.push_back(row);
				row += 10;
			}
			//cout << row << " " << (int)(prob.at<uchar>(row, col)) << endl;
		}
		if (CanBeIndex.size() >= 2)
		{
			int MeanIndex = CanBeIndex[CanBeIndex.size() - 2];
			int BoneIndex = CanBeIndex[CanBeIndex.size() - 1];
			MeatBounding.push_back(MeanIndex);
			BoneBounding.push_back(BoneIndex);

			nonZeroIndex.push_back(col - 60);
		}
		//else /*if (CanBeIndex.size() == 0)*/
		//{
		//	MeatBounding.push_back(-1);
		//	BoneBounding.push_back(-1);
		//}
	}
	#pragma endregion
	#pragma region 對應到 World Coordinate
	// 產生 Array
	int size = nonZeroIndex.size();
	float* _PointData = new float[size * 2 * 2]; // xy, 上下
	float** PointData = new float*[size * 2];
	for (int i = 0; i < size * 2; i++)
		PointData[i] = &_PointData[2 * i];
	memset(_PointData, 0, sizeof(float) * size * 2 * 2);

	// 把資料塞進去
	for (int i = 0; i < nonZeroIndex.size(); i++)
	{
		int index = nonZeroIndex[i];
		PointData[i][0] = index + 60;
		PointData[i][1] = MeatBounding[i];

		PointData[size + i][0] = index + 60;
		PointData[size + i][1] = BoneBounding[i];
	}
	float** WorldPos = calibrationTool.Calibrate(PointData, size * 2, 2);

	delete[] _PointData;
	delete[] PointData;
	#pragma endregion
	#pragma region 算世界座標的距離
	DistanceBounding.clear();
	/*DistanceMax = 0;
	DistanceMin = 100;*/
	DistanceMax = 6;
	DistanceMin = 3;
	for (int i = 0; i < nonZeroIndex.size(); i++)
	{
		QVector2D MeatPoint, BonePoint;
		MeatPoint.setX(WorldPos[i][0]);
		MeatPoint.setY(WorldPos[i][1]);

		BonePoint.setX(WorldPos[size + i][0]);
		BonePoint.setY(WorldPos[size + i][1]);
		
		DistanceBounding.push_back(MeatPoint.distanceToPoint(BonePoint));
		//cout << DistanceBounding[i] << endl;

		/*if (DistanceBounding[i] > DistanceMax)
			DistanceMax = DistanceBounding[i];
		if (DistanceBounding[i] < DistanceMin)
			DistanceMin = DistanceBounding[i];*/
	}
	#pragma endregion
	#pragma region 畫上結果
	Mat ColorMap = imread("./Images/ColorMap.png", IMREAD_COLOR);
	Mat ColorMapDepth = Mat::zeros(Size(250, 250), CV_8UC3);
	for (int i = 0; i < nonZeroIndex.size(); i++)
	{
		cv::Point p1, p2;
		p1.x = nonZeroIndex[i] + 60;
		p1.y = MeatBounding[i];

		p2.x = nonZeroIndex[i] + 60;
		p2.y = BoneBounding[i];

		float rate = (DistanceBounding[i] - DistanceMin) / (DistanceMax - DistanceMin);
		//cout << rate << endl;
		int GetRowIndex = ColorMap.rows * (1 - rate);
		int GetColIndex = ColorMap.cols * 0.5;
		Scalar color = ColorMap.at<Vec3b>(GetRowIndex, GetColIndex);
		line(ColorMapDepth, p1, p2, color);
	}
	bitwise_not(prob.clone(), prob);
	#pragma endregion 
	#pragma region 轉成 QOpenGLTexture
	// 轉 QOpenGLtexture
	OtherSideTexture = new QOpenGLTexture(Mat2QImage(otherSide, CV_8UC3));
	ProbTexture = new QOpenGLTexture(Mat2QImage(prob, CV_8UC3));
	DepthTexture = new QOpenGLTexture(Mat2QImage(ColorMapDepth, CV_8UC3));
	#pragma endregion
	#pragma region UI Max Min Value
	MaxValueLabel->setText(QString::number(DistanceMax));
	MinValueLabel->setText(QString::number(DistanceMin));
	#pragma endregion
}
float OpenGLWidget::GetDistanceValue(int index)
{
	#pragma region 先判斷是否有東西要跳出
	if (DistanceBounding.size() == 0)
		return -1;
	#pragma endregion
	#pragma region 接著是去跑結果
	return 10;
	#pragma endregion
}

// Helper Function
QImage OpenGLWidget::Mat2QImage(cv::Mat const& src, int Type)
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
Mat OpenGLWidget::GetBoundingBox(Mat img, QVector2D& TopLeft, QVector2D& ButtomRight)
{
	#pragma region 根據閘值，去抓邊界
	// 先根據閘值，並抓取邊界
	Mat threshold_output;
	vector<vector<cv::Point> > contours;
	vector<Vec4i> hierarchy;
	threshold(img, threshold_output, 8, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	// 先給占存 Array
	vector<BoundingBoxDataStruct> dataInfo(contours.size());
	#pragma endregion
	#pragma region 抓出最大的框框
	// 抓出擬合的結果
	for (size_t img = 0; img < contours.size(); img++)
	{
		BoundingBoxDataStruct data;
		approxPolyDP(Mat(contours[img]), data.contoursPoly, 3, true);
		data.boundingRect = boundingRect(Mat(data.contoursPoly));

		// 加進陣列
		dataInfo[img] = data;
	}
	Mat drawing = img.clone();
	cvtColor(drawing.clone(), drawing, CV_GRAY2BGR);
	sort(dataInfo.begin(), dataInfo.end(), SortByContourPointSize);

	// 抓出最亮，且最大的
	int i = 0;
	vector<vector<cv::Point>> contoursPoly(1);
	contoursPoly[0] = dataInfo[i].contoursPoly;

	// 邊界
	cv::Point tl = dataInfo[i].boundingRect.tl();
	tl.x = max(0, tl.x);
	tl.y = max(0, tl.y);
	cv::Point br = dataInfo[i].boundingRect.br();
	br.x = min(1024, br.x);
	br.y = min(250, br.y);

	rectangle(drawing, tl, br, Scalar(0, 255, 255), 2, 8, 0);

	TopLeft.setX(tl.x);
	TopLeft.setY(tl.y);
	ButtomRight.setX(br.x);
	ButtomRight.setY(br.y);
	#pragma endregion
	return drawing;
}
bool OpenGLWidget::SortByContourPointSize(BoundingBoxDataStruct& c1, BoundingBoxDataStruct& c2)
{
	return c1.boundingRect.area() > c2.boundingRect.area();
}