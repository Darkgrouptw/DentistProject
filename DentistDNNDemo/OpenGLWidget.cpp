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

		Program->setUniformValue("tempuv", tempuv);

		OtherSideTexture->bind(0);
		ProbTexture->bind(1);
		DepthTexture->bind(2);

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		OtherSideTexture->release();
		ProbTexture->release();
		DepthTexture->release();
		Program->release();

		//DrawSlider();



		glLineWidth(2.0f);
		glPushMatrix();
		glBegin(GL_LINES);

		//glColor3f(163.0 / 255.0, 126.0 / 255.0, 204.0 / 255.0);
		glColor3f(1.0, 0.0, 0.0);

		for (float i = -1.0; i < 1.0; i += 0.1) {
			glVertex2f(i, -temp);
			glVertex2f(i + 0.035, -temp);
		}

		glVertex2f(tempnn+ 0.05, -temp + 0.05);
		glVertex2f(tempnn- 0.05, -temp- 0.05);
		glVertex2f(tempnn+ 0.05, -temp- 0.05);
		glVertex2f(tempnn - 0.05, -temp + 0.05);


		
		glEnd();
		glPopMatrix();

		glLineWidth(4.0f);

		glPushMatrix();
		glBegin(GL_LINES);

		glColor3f(250.0 / 255.0, 220.0 / 255.0, 0.0 / 255.0);
		glVertex2f(tempnn, -1.0);
		glVertex2f(tempnn, 1.0);

		glEnd();
		glPopMatrix();

		glColor3f(1.0, 1.0, 1.0);
	}


}
void OpenGLWidget::DrawSlider() {
	if (CheckIsNonZeroValue) {
		glColor3f(1.0, 0.0, 0.0);
		glLineWidth(5.0f);
		glBegin(GL_LINES);

		glVertex3f((WorldPosMeat[SliderValue].y() / 250.0f - 0.5f) * 2.0f + 0.05f, -(WorldPosMeat[SliderValue].x() / 250.0f - 0.5f) * 2.0f + 0.05f, 0.0f);
		glVertex3f((WorldPosMeat[SliderValue].y() / 250.0f - 0.5f) * 2.0f - 0.05f, -(WorldPosMeat[SliderValue].x() / 250.0f - 0.5f) * 2.0f - 0.05f, 0.0f);
		glVertex3f((WorldPosMeat[SliderValue].y() / 250.0f - 0.5f) * 2.0f + 0.05f, -(WorldPosMeat[SliderValue].x() / 250.0f - 0.5f) * 2.0f - 0.05f, 0.0f);
		glVertex3f((WorldPosMeat[SliderValue].y() / 250.0f - 0.5f) * 2.0f - 0.05f, -(WorldPosMeat[SliderValue].x() / 250.0f - 0.5f) * 2.0f + 0.05f, 0.0f);

		glVertex3f((WorldPosBone[SliderValue].y() / 250.0f - 0.5f) * 2.0f + 0.05f, -(WorldPosBone[SliderValue].x() / 250.0f - 0.5f) * 2.0f + 0.05f, 0.0f);
		glVertex3f((WorldPosBone[SliderValue].y() / 250.0f - 0.5f) * 2.0f - 0.05f, -(WorldPosBone[SliderValue].x() / 250.0f - 0.5f) * 2.0f - 0.05f, 0.0f);
		glVertex3f((WorldPosBone[SliderValue].y() / 250.0f - 0.5f) * 2.0f + 0.05f, -(WorldPosBone[SliderValue].x() / 250.0f - 0.5f) * 2.0f - 0.05f, 0.0f);
		glVertex3f((WorldPosBone[SliderValue].y() / 250.0f - 0.5f) * 2.0f - 0.05f, -(WorldPosBone[SliderValue].x() / 250.0f - 0.5f) * 2.0f + 0.05f, 0.0f);

		glEnd();
	}
	glColor3f(1.0, 1.0, 1.0);
}

// 外部呼叫函式
void OpenGLWidget::ProcessImg(Mat otherSide, Mat prob, QVector<Mat> FullMat, QVector2D OriginTL, QVector2D OriginBR, QLabel* MaxValueLabel, QLabel* MinValueLabel)
{
	#pragma region 生成BoneMap與MeatMap
	MeatMap = prob.clone();
	threshold(MeatMap.clone(), MeatMap, 150, 255, THRESH_BINARY);
	bitwise_not(MeatMap.clone(), MeatMap);
	thin(MeatMap, false, false, false);
	cvtColor(MeatMap.clone(), MeatMap, CV_GRAY2BGR);

	BoneMap = cv::Mat::zeros(prob.size(), CV_8UC3);
	bitwise_not(BoneMap.clone(), BoneMap);
	#pragma endregion	
	#pragma region 算方向的結果
	Mat MomentMeat;
	#pragma endregion
	#pragma region 反轉顏色
	threshold(prob.clone(), prob, 150, 255, THRESH_BINARY);

	bitwise_not(prob.clone(), prob);
	thin(prob, false, false, false);

	// 存在這邊
	MomentMeat = prob.clone();
	bitwise_not(MomentMeat.clone(), MomentMeat);
	//imwrite("D:/a.png", MomentMeat);
	#pragma endregion
	#pragma region OtherSide Clone
	cvtColor(otherSide.clone(), otherSide, CV_GRAY2BGR);
	cvtColor(prob.clone(), prob, CV_GRAY2BGR);

	if (OtherSideTexture != NULL)
	{
		delete OtherSideTexture;
		delete DepthTexture;
		delete ProbTexture;
	}
	#pragma endregion
	#pragma region 抓出結果
	// 每一張圖片的結果
	vector<Mat> ChannelMat;
	QVector2D TL, BR;
	QVector<int> LastTLY;
	QVector<int> LastBLY;
	//int LastY = -1;
	for (int i = 0; i < FullMat.size(); i++)
	{
		// 抓取藍色的部分，去抓取 Bounding Box
		split(FullMat[i], ChannelMat);

		// 刪除一些 Noise
		vector<vector<cv::Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(ChannelMat[0], contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		sort(contours.begin(), contours.end(), CompareContourArea);

		for (int k = 1; k < contours.size(); k++)
			drawContours(ChannelMat[0], contours, k, Scalar(0, 0, 0), CV_FILLED);
		//imwrite("E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 3.2/Smooth/Smooth_" + to_string(i) + ".png", ChannelMat[0]);

		Mat img = GetBoundingBox(ChannelMat[0], TL, BR);

		if (TL.y() == 0 && BR.y() == ChannelMat[0].cols) {
			LastTLY.append(-1);
			LastBLY.append(-1);
		}
		else {
			LastTLY.append(TL.y() + OriginTL.y());
			LastBLY.append(BR.y() + OriginTL.y());
		}
	}
	
	// 內插出沒有齒槽骨區塊
	for (int i = 1; i < LastBLY.size(); i++) {
		int tempTL = 0;
		int tempBL = 0;
		int num = 0;		// 間隔

		if (LastBLY[i] == -1 && LastTLY[i] == -1) {
			for (int temp = 1; temp < 20; temp++) {
				if (LastBLY[i + temp] != -1 && LastTLY[i + temp] != -1 && (temp + i) < LastBLY.size()) {
					tempTL = LastTLY[i + temp];
					tempBL = LastBLY[i + temp];
					num = temp;
					break;
				}
			}
			if (num == 0) {
				if (LastTLY[i - 1] != -1 && LastBLY[i - 1] != -1) {
					LastTLY[i] = LastTLY[i - 1];
					LastBLY[i] = LastBLY[i - 1];
				}
			}
			else {
				LastTLY[i] = LastTLY[i - 1] + (tempTL - LastTLY[i - 1]) / num;
				LastBLY[i] = LastBLY[i - 1] + (tempBL - LastBLY[i - 1]) / num;
			}
		}
	}

	// Smooth Point
	for (int i = 0; i < LastBLY.size(); i++) {
		if (LastBLY[i] != -1 && LastTLY[i] != -1) {
			int tempTL = 0;
			int tempBR = 0;
			for (int j = -SmoothRange; j <= SmoothRange; j++) {
				if (((i + j) < 0) || (i + j >= LastBLY.size())) {
					tempTL += LastTLY[i];
					tempBR += LastBLY[i];
					continue;
				}
				tempTL += LastTLY[i + j];
				tempBR += LastBLY[i + j];
			}
			LastTLY[i] = tempTL / (SmoothRange * 2 + 1);
			LastBLY[i] = tempBR / (SmoothRange * 2 + 1);
		}
	}
	#pragma endregion
	#pragma region 畫上齒槽骨結果
	IsInverse = false;				// 是否上下顛倒
	int LastSize = LastTLY.size();
	if (LastTLY[0] < 125 && LastTLY[LastSize / 2] < 125 && LastTLY[LastSize - 1] < 125)
		IsInverse = true;

	// 畫結果
	for (int i = 1; i < LastTLY.size(); i++)
	{
		cv::Point LeftPoint = cv::Point(i - 1 + 60, LastTLY[i - 1]);
		cv::Point RightPoint = cv::Point(i + 60, LastTLY[i]);

		if (IsInverse)
		{
			LeftPoint.y = LastBLY[i - 1];
			RightPoint.y = LastBLY[i];
		}

		if (LastTLY[i - 1] != -1 && LastTLY[i] != -1) {
			line(prob, LeftPoint, RightPoint, Scalar(0, 0, 0), 1);
			line(BoneMap, LeftPoint, RightPoint, Scalar(0, 0, 0), 1);
		}
	}
	//imshow("Bonemap", BoneMap);
	//waitKey(0);
	bitwise_not(BoneMap.clone(), BoneMap);
	imwrite("D:/N.png", BoneMap);
	bitwise_not(BoneMap.clone(), BoneMap);

	#pragma endregion
	#pragma region 算 Moment (也就是距離的方向)
	cv::Moments m = cv::moments(MomentMeat, true);

	double U20 = m.nu20 / m.m00;
	double U02 = m.nu02 / m.m00;
	double U11 = m.nu11 / m.m00;
	double Angle = 1.5 * 0.5 * atan(2 * U11 / (U20 - U11));
	Angle = (!IsInverse ? Angle + 3.1415926 / 2 : Angle - 3.1415926 / 2);			// 由於坐標系不一樣，需要做一個轉換
	//cout << Angle << endl;															// 是徑度喔
	float slope = tan(Angle);
	//cout << slope << endl;
	#pragma endregion
	#pragma region 算出牙肉 & 齒槽骨的位置(Pixel)
	WorldPosMeat.clear();
	WorldPosBone.clear();

	// 算出牙肉所有位置
	for (int row = 0; row < MeatMap.rows; row++) {
		//for (int col = 0; col < MeatMap.cols; col++) {
		for (int col = 60; col <= 200; col++) {
			if (MeatMap.at<Vec3b>(row, col)[0] == 0 && MeatMap.at<Vec3b>(row, col)[1] == 0 && MeatMap.at<Vec3b>(row, col)[2] == 0) {
				WorldPosMeat.push_back(QVector2D(row, col));
				//cout << row << " " << col << endl;
			}
		}
	}
	//cout << WorldPosMeat.size() << endl;

	// 牙肉 row += sin(Angle)	的方式前進尋找齒槽骨
	//		col += cos(Angle)
	for (int i = 0; i < WorldPosMeat.size(); i++) {
		int row = WorldPosMeat[i].x();
		int col = WorldPosMeat[i].y();
		bool isAdd = false;
		WorldPosBone.push_back(QVector2D(-1, -1));

		for (int count = 0; count < 250; count++) {
			int newrow = row + count * sin(Angle);
			int newcol = col + count * cos(Angle);
			if (newrow > 248 || newrow < 2 || newcol > 248 || newcol < 2) break;

			if (BoneMap.at<Vec3b>(newrow, newcol)[0] == 0) {
				WorldPosBone[i] = QVector2D(newrow, newcol);
				nonZeroIndex.push_back(i);
				isAdd = true;
				break;
			}
		}
	}
	/*
	// 牙肉往斜率(m)方向前進找齒槽骨
	for (int i = 0; i < WorldPosMeat.size(); i++) {
		int row = WorldPosMeat[i].x();
		int col = WorldPosMeat[i].y();
		bool isAdd = false;

		for (int j = 0; j < BoneMap.cols; j++) {
			if (abs(slope) >= 5) {
				if (!IsInverse ? (row + j) >= 248 : (row - j) <= 2)break;
				if (BoneMap.at<Vec3b>(!IsInverse ? (row + j) : (row - j), col)[0] == 0) {
					WorldPosBone.push_back(QVector2D(!IsInverse ? (row + j) : (row - j), col));
					nonZeroIndex.push_back(i);
					isAdd = true;
					break;
				}
			}
			else
			{
				if (!IsInverse ? (row + j * slope) >= 248 : (row + j * slope) <= 2)break;
				for (int mRange = -ceil(abs(slope / 2.0f)); mRange <= ceil(abs(slope / 2.0f)); mRange++) {
					if (BoneMap.at<Vec3b>(floor(row + j * slope) + mRange, col + j)[0] == 0) {
						WorldPosBone.push_back(QVector2D(floor(row + j * slope) + mRange, col + j));
						nonZeroIndex.push_back(i);
						isAdd = true;
						break;
					}
				}
			}
			if (isAdd)break;
		}
		//cout << row << " " << col << endl;
	}
	//imwrite("D:/BoneMap.png", BoneMap);
	//imwrite("D:/MeatMap.png", MeatMap);
	*/
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

		PointData[i][0] = WorldPosMeat[index].y();
		PointData[i][1] = WorldPosMeat[index].x();

		PointData[size + i][0] = WorldPosBone[index].y();
		PointData[size + i][1] = WorldPosBone[index].x();
		//cout << PointData[i][0] << " " << PointData[i][1] << " " << PointData[size + i][0] << " " << PointData[size + i][1] << endl;
	}
	float** WorldPos = calibrationTool.Calibrate(PointData, size * 2, 2);

	delete[] _PointData;
	delete[] PointData;
	#pragma endregion
	#pragma region 算世界座標的距離
	DistanceBounding.clear();
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

		if (DistanceBounding[i] > DistanceMax)
			DistanceMax = DistanceBounding[i];
		if (DistanceBounding[i] < DistanceMin) 
			DistanceMin = DistanceBounding[i];
	}
	#pragma endregion
	#pragma region 畫上結果
	Mat ColorMap = imread("./Images/ColorMap.png", IMREAD_COLOR);
	Mat ColorMapDepth = Mat::zeros(Size(250, 250), CV_8UC3);
	bitwise_not(prob.clone(), prob);

	// 畫上線的顏色
	for (int i = 0; i < nonZeroIndex.size(); i++)
	{
		int index = nonZeroIndex[i];

		float rate = (DistanceBounding[i] - DistanceMin) / (DistanceMax - DistanceMin);
		if (0 >= rate) rate = 0;
		if (1 <= rate) rate = 1;

		int GetRowIndex = ColorMap.rows * (1 - rate);
		int GetColIndex = ColorMap.cols * 0.5;
		//prob.at<Vec3b>(WorldPosMeat[index].x(), WorldPosMeat[index].y()) = ColorMap.at<Vec3b>(GetRowIndex, GetColIndex);
	}
	//imwrite("D:/res.png", prob);
	#pragma endregion
	#pragma region 轉成 QOpenGLTexture

	bitwise_not(MeatMap.clone(), MeatMap);
	bitwise_not(BoneMap.clone(), BoneMap);

	for (int row = 0; row < BoneMap.rows; row++) {
		for (int col = 0; col < BoneMap.cols; col++) {
			if (BoneMap.at<Vec3b>(row, col) == Vec3b(255, 255, 255))
				BoneMap.at<Vec3b>(row, col) = Vec3b(204, 126, 163);
		}
	}

	// 轉 QOpenGLtexture
	OtherSideTexture = new QOpenGLTexture(Mat2QImage(otherSide, CV_8UC3));
	ProbTexture = new QOpenGLTexture(Mat2QImage(BoneMap, CV_8UC3));
	DepthTexture = new QOpenGLTexture(Mat2QImage(MeatMap, CV_8UC3));
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
float OpenGLWidget::GetSliderValue(int value)
{
	/*
	CheckIsNonZeroValue = false;
	for (int i = 0; i < nonZeroIndex.size(); i++) {
		int index = nonZeroIndex[i];

		if (value == WorldPosMeat[index].y()) {
			CheckIsNonZeroValue = true;
			SliderValue = index;
			break;
		}
	}
	*/
	
	for (int row = 0; row < BoneMap.rows; row++)
		if (BoneMap.at<Vec3b>(row, value) != Vec3b(0, 0, 0))
			SliderValue = ((row / 250.0f) - 0.5f) * 2.0f;		// Slider現在的位置(-1 ~ 1)
			//SliderValue = row;

	//cout << SliderValue << endl;

	tempuv = (value / 250.0f);
	temp = SliderValue;
	tempnn= ((value / 250.0f) - 0.5f) * 2.0f;

	//cout << tempuv << endl;

	return temp;
}
QString OpenGLWidget::GetColorMapValue(int value) {
	QString rate = 0;
	for (int i = 0; i < nonZeroIndex.size(); i++) {
		int index = nonZeroIndex[i];

		if (value == WorldPosMeat[index].y()) {
			rate = QString::fromStdString(to_string(DistanceBounding[i] - 0.5f));
			break;
		}
	}
	return rate;
}
int OpenGLWidget::GetNowSliderValue(int value) {
	int num = -1;
	for (int i = 0; i < nonZeroIndex.size(); i++) {
		int index = nonZeroIndex[i];

		if (value == WorldPosMeat[index].y()) {
			num = WorldPosMeat[i].x();
			break;
		}
	}
	return num;
}

void OpenGLWidget::TestWriteDistance(Mat Org, Mat Prob)
{
	const int needlerow = 92;
	const int needlecol = 121;
#pragma region 生成VaildMap
	cv::Mat VaildMap = Prob.clone();
	threshold(VaildMap.clone(), VaildMap, 150, 255, THRESH_BINARY);
	bitwise_not(VaildMap.clone(), VaildMap);
	thin(VaildMap, false, false, false);
	cvtColor(VaildMap.clone(), VaildMap, CV_GRAY2BGR);

	cvtColor(Org.clone(), Org, CV_GRAY2BGR);
#pragma endregion	
#pragma region 將位置放入矩陣
	QVector<QVector2D> MeatPos;
	QVector<QVector2D> TestPos;

	 for (int row = 0; row < VaildMap.rows; row++) {
		 for (int col = 0; col < VaildMap.cols; col++) {
			 if (VaildMap.at<Vec3b>(row, col)[0] == 0) {
				 MeatPos.push_back(QVector2D(needlerow, needlecol));
				 TestPos.push_back(QVector2D(row, col));
				 //cout << "Meat :" << needlerow << " " << needlecol << endl;
				 //cout << "Test :" << row << " " << col << endl;
			 }
		 }
	 }
#pragma endregion	
#pragma region 對應到 World Coordinate
	// 產生 Array
	int size = MeatPos.size();
	float* _PointData = new float[size * 2 * 2]; // xy, 上下
	float** PointData = new float*[size * 2];
	for (int i = 0; i < size * 2; i++)
		PointData[i] = &_PointData[2 * i];
	memset(_PointData, 0, sizeof(float) * size * 2 * 2);

	// 把資料塞進去
	for (int i = 0; i < MeatPos.size(); i++)
	{
		PointData[i][0] = MeatPos[i].y();
		PointData[i][1] = MeatPos[i].x();

		PointData[size + i][0] = TestPos[i].y();
		PointData[size + i][1] = TestPos[i].x();
	}
	float** WorldPos = calibrationTool.Calibrate(PointData, size * 2, 2);

	delete[] _PointData;
	delete[] PointData;
#pragma endregion
#pragma region 算世界座標的距離
	DistanceBounding.clear();
	DistanceMax = 6;
	DistanceMin = 3;
	QVector<int> CanbeOneMM;
	QVector<int> CanbeTwoMM;

	for (int i = 0; i < MeatPos.size(); i++)
	{
		QVector2D MeatPoint, BonePoint;
		MeatPoint.setX(WorldPos[i][0]);
		MeatPoint.setY(WorldPos[i][1]);

		BonePoint.setX(WorldPos[size + i][0]);
		BonePoint.setY(WorldPos[size + i][1]);

		DistanceBounding.push_back(MeatPoint.distanceToPoint(BonePoint));

		if (DistanceBounding[i] > DistanceMax) DistanceMax = DistanceBounding[i];
		if (DistanceBounding[i] < DistanceMin) DistanceMin = DistanceBounding[i];
		//cout << DistanceBounding[i] << endl;
	}

	vector<float> vecDistance = DistanceBounding.toStdVector();
	sort(vecDistance.begin(), vecDistance.end());
	auto const Onemm = std::lower_bound(vecDistance.begin(), vecDistance.end(), 1);
	auto const Twomm = std::lower_bound(vecDistance.begin(), vecDistance.end(), 2);

	for (int i = 0; i < DistanceBounding.size(); i++) {
		//cout << DistanceBounding[i] << endl;
		if (DistanceBounding[i] == *Onemm)CanbeOneMM.push_back(i);
		if (DistanceBounding[i] == *Twomm)CanbeTwoMM.push_back(i);
	}
#pragma endregion
	for (int i = -3; i <= 3; i++) {
		Org.at<Vec3b>(TestPos[CanbeOneMM[0]].x() + i, TestPos[CanbeOneMM[0]].y() + i) = Vec3b(255, 255, 255);
		Org.at<Vec3b>(TestPos[CanbeOneMM[0]].x() - i, TestPos[CanbeOneMM[0]].y() + i) = Vec3b(255, 255, 255);

		Org.at<Vec3b>(TestPos[CanbeTwoMM[0]].x() + i, TestPos[CanbeTwoMM[0]].y() + i) = Vec3b(255, 255, 255);
		Org.at<Vec3b>(TestPos[CanbeTwoMM[0]].x() - i, TestPos[CanbeTwoMM[0]].y() + i) = Vec3b(255, 255, 255);

	}
	//imwrite("D:/OtherSideOrg.png", Org);
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
	if (contours.size() == 0)
	{
		TopLeft = QVector2D(0, 0);
		ButtomRight = QVector2D(img.rows, img.cols);
		return img;
	}

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
bool OpenGLWidget::CompareContourArea(vector<cv::Point> contour1, vector<cv::Point> contour2)
{
	// comparison function object
	double i = fabs(contourArea(Mat(contour1)));
	double j = fabs(contourArea(Mat(contour2)));
	return i > j;
}
