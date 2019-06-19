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
		OtherSideTexture->bind(0);
		ProbTexture->bind(1);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		OtherSideTexture->release();
		ProbTexture->release();
		Program->release();
	}
}

// 外部呼叫函式
void OpenGLWidget::ProcessImg(Mat otherSide, Mat prob, QVector<Mat> FullMat, QVector2D OriginTL, QVector2D OriginBR)
{
	#pragma region 反轉顏色
	threshold(prob.clone(), prob, 150, 255, THRESH_BINARY);

	bitwise_not(prob.clone(), prob);
	thin(prob, false, false, false);
	#pragma endregion
	#pragma region 算出牙肉位置(Pixel)
	for (int col = 60; col <= 200; col++)
		for (int row = prob.rows - 1; row >= 0; row--)
			if (prob.at<uchar>(row, col) == 0) {
				MeatBounding.push_back(row);
				break;
			}
			else if (row == 0)MeatBounding.push_back(-1);
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
	#pragma region 算出齒槽骨位置(Pixel)
	for (int col = 60; col <= 200; col++)
		for (int row = prob.rows - 1; row >= 0; row--)
			if (prob.at<Vec3b>(row, col)[0] == 0) {
				DiseaseBounding.push_back(row);
				break;
			}
			else if (row == MeatBounding[col - 60])DiseaseBounding.push_back(-1);
	#pragma endregion
	#pragma region 轉成 QOpenGLTexture
	// 轉 QOpenGLtexture
	OtherSideTexture = new QOpenGLTexture(Mat2QImage(otherSide, CV_8UC3));
	ProbTexture = new QOpenGLTexture(Mat2QImage(prob, CV_8UC3));
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