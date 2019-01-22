#include "OpenGLWidget.h"

OpenGLWidget::OpenGLWidget(QWidget* parent = 0) : QOpenGLWidget(parent)
{
}
OpenGLWidget::~OpenGLWidget()
{
	if (program != NULL)
		delete program;
}

void OpenGLWidget::initializeGL()
{
	initializeOpenGLFunctions();
	glClearColor(0.5f, 0.5, 0.5f, 1);
	glEnable(GL_DEPTH_TEST);

	InitProgram();
	CalcMatrix();
}
void OpenGLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glLoadMatrixf((ProjectionMatrix * ViewMatrix).data());

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	DrawGround();
	

	if (RotationMode)
		DrawResetRotation();
	else
	{
		DrawPointCloud();
		DrawSTL();
	}
}

//////////////////////////////////////////////////////////////////////////
// 滑鼠事件
//////////////////////////////////////////////////////////////////////////
void OpenGLWidget::mousePressEvent(QMouseEvent *event)
{
	PressPoint = event->pos();
	TempArcAngle = ArcAngle;
	TempElevationAngle = ElevationAngle;
}
void OpenGLWidget::mouseMoveEvent(QMouseEvent *event)
{
	CurrentPoint = event->pos();

	int width = CurrentPoint.x() - PressPoint.x();
	float rate = (float)width / this->width();
	ArcAngle = TempArcAngle + rate * 180;

	//邊界判定
	if (ArcAngle >= 360)
		ArcAngle -= 360;
	if (ArcAngle <= 360)
		ArcAngle += 360;
	
	int height = CurrentPoint.y() - PressPoint.y();
	rate = (float)height / this->height();
	ElevationAngle = qBound(-89, TempElevationAngle + (int)(rate * 180), 89);

	// 更新 Widget
	CalcMatrix();
	this->update();
}
void OpenGLWidget::wheelEvent(QWheelEvent *event)
{
	// Most mouse types work in steps of 15 degrees, in which case the delta value is a multiple of 120; i.e., 120 units * 1/8 = 15 degrees
	// (From: http://doc.qt.io/qt-5/qwheelevent.html#angleDelta)
	int degree = event->angleDelta().y() / 8;
	int moveRadius = -degree / RadiusSpeed_Dev;

	Radius = qBound(MinRadius, Radius + moveRadius, MaxRadius);

	// 更新 Widget
	CalcMatrix();
	this->update();
}

//////////////////////////////////////////////////////////////////////////
// 外部呼叫函數
//////////////////////////////////////////////////////////////////////////
bool OpenGLWidget::LoadSTLFile(QString FileName)
{
	#pragma region 讀取 STL 檔案 & 設定
	if (!OpenMesh::IO::read_mesh(STLFile, FileName.toStdString()))
	{
		cout << "讀取錯誤!!" << endl;
		IsLoaded = false;
		return false;
	}
	cout << "模型面數：" << STLFile.n_faces() << endl;
	IsLoaded = true;

	// Add property
	srand(time(NULL));
	PointArray.clear();
	STLFile.add_property(AreaInfo, "AreaInfo");
	#pragma endregion
	#pragma region 找出 Bounding Box
	float maxX, maxY, maxZ;
	float minX, minY, minZ;
	maxX = maxY = maxZ = -99999;
	minX = minY = minZ = 99999;

	// 跑每一個點
	for (MeshType::VertexIter v_iter = STLFile.vertices_begin(); v_iter != STLFile.vertices_end(); v_iter++)
	{
		MeshType::Point p = STLFile.point(v_iter);
		#pragma region X 判斷
		if (p[0] < minX)
			minX = p[0];
		if (p[0] > maxX)
			maxX = p[0];
		#pragma endregion
		#pragma region Y 判斷
		if (p[1] < minY)
			minY = p[1];
		if (p[1] > maxY)
			maxY = p[1];
		#pragma endregion
		#pragma region Z 判斷
		if (p[2] < minZ)
			minZ = p[2];
		if (p[2] > maxZ)
			maxZ = p[2];
		#pragma endregion
	}

	cout << "Bounding Box" << endl;
	cout << "Max: " << maxX << " " << maxY << " " << maxZ << endl;
	cout << "Min: " << minX << " " << minY << " " << minZ << endl;

	BoundingBox[0] = QVector3D(maxX, maxY, maxZ);
	BoundingBox[1] = QVector3D(minX, minY, minZ);
	#pragma endregion
	#pragma region 找出 Transform Matrix
	// Reset Matrix
	TransformMatrix.setToIdentity();
	float deltaX = maxX - minX;
	float deltaY = maxY - minY;
	float deltaZ = maxZ - minZ;

	// 這邊代表要旋轉
	if (deltaY > deltaX || deltaY > deltaZ)
	{
		TransformMatrix.rotate(QQuaternion::fromEulerAngles(QVector3D(-90, 0, 0)));
		cout << "旋轉模型" << endl;
	}

	// 這邊再算 scale，只有 deltaZ 要乘 2，因為 Z 只有 0 ~ 10 (其他 -10 ~ 10)
	float maxDelta;
	if (deltaX > deltaY && deltaX > deltaZ * 2)
		maxDelta = deltaX;
	else if (deltaY > deltaX && deltaY > deltaZ * 2)
		maxDelta = deltaY;
	else
		maxDelta = deltaZ * 2;
	cout << "Delta: " << deltaX << " " << deltaY << " " << deltaZ << endl;

	float scale = GridSize / maxDelta * 2;					// 因為是 -10 ~ 10，所以是 20
	TransformMatrix.scale(QVector3D(scale, scale, scale));
	cout << "Scale " << scale << endl;

	// 算平面的結果 (X, Y, Z 平移)
	QVector4D squareMaxPos(maxX, maxY, maxZ, 1);
	QVector4D squareMinPos(minX, minY, minZ, 1);
	squareMaxPos = TransformMatrix * squareMaxPos;
	squareMinPos = TransformMatrix * squareMinPos;

	cout << "After Transform" << endl;
	cout << "Max: " << squareMaxPos.x() << " " << squareMaxPos.y() << " " << squareMaxPos.z() << endl;
	cout << "Min: " << squareMinPos.x() << " " << squareMinPos.y() << " " << squareMinPos.z() << endl;

	if (deltaY > deltaX || deltaY > deltaZ)
	{
		float offsetX = (squareMaxPos.x() + squareMinPos.x()) / 2;
		float offsetZ = (squareMaxPos.z() + squareMinPos.z()) / 2;
		OffsetToCenter = QVector3D(-offsetX, -squareMinPos.y(), -offsetZ);
	}
	else
	{
		float offsetX = (squareMaxPos.x() + squareMinPos.x()) / 2;
		float offsetZ = (squareMaxPos.z() + squareMinPos.z()) / 2;
		OffsetToCenter = QVector3D(-offsetX, -squareMinPos.y(), -offsetZ);
	}

	QVector4D tempOffset = TransformMatrix.inverted() * QVector4D(OffsetToCenter, 1);
	TransformMatrix.translate(tempOffset.toVector3D());
	cout << "Offset: " << OffsetToCenter.x() << " " << OffsetToCenter.y() << " " << OffsetToCenter.z() << endl;
	#pragma endregion
	#pragma region 跑每一個點，算出面積
	QVector<float> EdgeLength;
	float AreaTotal = 0;
	for (MeshType::FaceIter f_it = STLFile.faces_begin(); f_it != STLFile.faces_end(); f_it++)
	{
		EdgeLength.clear();
		for (MeshType::FaceEdgeIter fe_it = STLFile.fe_iter(f_it); fe_it.is_valid(); fe_it++)
		{
			float length = STLFile.calc_edge_length(fe_it.handle());
			EdgeLength.push_back(length);
		}
		float currentArea = CalcArea(EdgeLength);
		STLFile.property(AreaInfo, f_it) = currentArea;
		AreaTotal += currentArea;
	}
	cout << "面積: " << AreaTotal << endl;

	// 開始撒點
	QVector<QVector3D> TempPointArray;
	float remainArea = 0;
	for (MeshType::FaceIter f_it = STLFile.faces_begin(); f_it != STLFile.faces_end(); f_it++)
	{
		float currentArea = STLFile.property(AreaInfo, f_it);
		TempPointArray.clear();
		for (MeshType::FaceEdgeIter fe_it = STLFile.fe_iter(f_it); fe_it.is_valid(); fe_it++)
		{
			MeshType::Point p = STLFile.point(STLFile.from_vertex_handle(STLFile.halfedge_handle(fe_it, 0)));

			QVector3D vecP(p[0], p[1], p[2]);
			TempPointArray.push_back(vecP);
		}

		// 數量
		float count_float = currentArea * SpreadingPointSize / AreaTotal;
		remainArea += count_float;
		int count = int(remainArea);
		remainArea -= count;

		for (int i = 0; i < count; i++)
			PointArray.push_back(SamplePoint(TempPointArray));
	}
	#pragma endregion
	#pragma region 跑每一個點，並算 BaryCentric 座標
	// 清空
	VertexData.clear();
	BaryCentricData.clear();

	// 結果
	for (MeshType::FaceIter f_it = STLFile.faces_begin(); f_it != STLFile.faces_end(); f_it++)
	{
		for (MeshType::FaceVertexIter fv_it = STLFile.fv_iter(f_it); fv_it.is_valid(); fv_it++)
		{
			// 取點
			MeshType::Point p = STLFile.point(fv_it);

			// 轉換成 QVector3D
			QVector3D tempPoint(p[0], p[1], p[2]);
			VertexData.push_back(tempPoint);
		}
		// 裝三個值進去
		BaryCentricData.push_back(QVector3D(1, 0, 0));
		BaryCentricData.push_back(QVector3D(0, 1, 0));
		BaryCentricData.push_back(QVector3D(0, 0, 1));
	}

	#pragma endregion
	#pragma region 產生 Buffer
	// 先刪除舊有的 Buffer
	if (VertexBuffer != -1)
	{
		glDeleteBuffers(1, &VertexBuffer);
		glDeleteBuffers(1, &BaryCentricBuffer);
	}

	// 產生 Buffer
	glGenBuffers(1, &VertexBuffer);
	glGenBuffers(1, &BaryCentricBuffer);

	// Bind Vertex Buffer
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, VertexData.size() * sizeof(QVector3D), VertexData.constData(), GL_STATIC_DRAW);

	// Bind BaryCentric Buffer
	glBindBuffer(GL_ARRAY_BUFFER, BaryCentricBuffer);
	glBufferData(GL_ARRAY_BUFFER, BaryCentricData.size() * sizeof(QVector3D), BaryCentricData.constData(), GL_STATIC_DRAW);
	#pragma endregion
	cout << PointArray.size() << endl;
	return true;
}
void OpenGLWidget::SetRenderTriangleBool(bool RenderBool)
{
	RenderTriangle_bool = RenderBool;
}
void OpenGLWidget::SetRenderBorderBool(bool RenderBool)
{
	RenderBorder_bool = RenderBool;
}
void OpenGLWidget::SetRenderPointCloudBool(bool RenderBool)
{
	RenderPointCloud_bool = RenderBool;
}
void OpenGLWidget::SetRotationMode(bool SetBool)
{
	RotationMode = SetBool;
	this->update();
}

//////////////////////////////////////////////////////////////////////////
// 其他元件的 Function
//////////////////////////////////////////////////////////////////////////
void OpenGLWidget::SetRawDataManager(RawDataManager* raw)
{
	this->rawManager = raw;
}

//////////////////////////////////////////////////////////////////////////
// 初始化
//////////////////////////////////////////////////////////////////////////
void OpenGLWidget::InitProgram()
{
	program = new QOpenGLShaderProgram();
	program->addShaderFromSourceFile(QOpenGLShader::Vertex,		"./Shaders/Model.vert");
	program->addShaderFromSourceFile(QOpenGLShader::Fragment,	"./Shaders/Model.frag");
	program->link();

	// Get Location
	ProjectionMatrixLoc		= program->uniformLocation("ProjectionMatrix");
	ViewMatrixLoc			= program->uniformLocation("ViewMatrix");
	ModelMatrixLoc			= program->uniformLocation("ModelMatrix");
}

//////////////////////////////////////////////////////////////////////////
// 矩陣相關
//////////////////////////////////////////////////////////////////////////
void OpenGLWidget::CalcMatrix()
{
	float AngleInRadian = ElevationAngle * M_PI / 180;
	float GroundRadius = Radius * cos(AngleInRadian);
	float Height = Radius * sin(AngleInRadian);

	float ArcAngleInRadian = ArcAngle * M_PI / 180;
	float GroundX = GroundRadius * cos(ArcAngleInRadian);
	float GroundZ = GroundRadius * sin(ArcAngleInRadian);

	ProjectionMatrix.setToIdentity();
	ProjectionMatrix.perspective(60, 1, 0.1f, 100);

	ViewMatrix.setToIdentity();
	ViewMatrix.lookAt(
		QVector3D(GroundX, Height, GroundZ),
		QVector3D(0, 0, 0),
		QVector3D(0, 1, 0)
	);
}
float OpenGLWidget::CalcArea(QVector<float> LengthArray)
{
	float a = LengthArray[0];
	float b = LengthArray[1];
	float c = LengthArray[2];
	float s = (a + b + c) / 2;
	return sqrt(s * (s - a) * (s - b) * (s - c));
}

QVector3D OpenGLWidget::SamplePoint(QVector<QVector3D> trianglePoints)
{
	QVector3D a = trianglePoints[0];
	QVector3D b = trianglePoints[1];
	QVector3D c = trianglePoints[2];

	float ra = (float)rand() / RAND_MAX / 2;
	float rb = (float)rand() / RAND_MAX / 2;
	float rc = 1 - ra - rb;
	return a * ra + b * rb + c *rc;
}

//////////////////////////////////////////////////////////////////////////
// 畫畫的 Function
//////////////////////////////////////////////////////////////////////////
void OpenGLWidget::DrawGround()
{
	#pragma region 地板
	float lowerY = -0.01f;
	glColor4f(0.7f, 0.7f, 0.7f, 1);
	glBegin(GL_QUADS);
	glVertex3f(GridMax.x(), lowerY, GridMax.y());
	glVertex3f(GridMin.x(), lowerY, GridMax.y());
	glVertex3f(GridMin.x(), lowerY, GridMin.y());
	glVertex3f(GridMax.x(), lowerY, GridMin.y());
	glEnd();
	#pragma endregion

	glColor4f(1.f, 1.f, 1.f, 1.f);
	glBegin(GL_LINES);
	#pragma region 橫
	for (int z = GridMin.y(); z <= GridMax.y(); z++)
	{
		glVertex3f(GridMin.x(), 0, z);
		glVertex3f(GridMax.x(), 0, z);
	}
	#pragma endregion
	#pragma region 直
	for (int x = GridMin.x(); x <= GridMax.x(); x++)
	{
		glVertex3f(x, 0, GridMin.y());
		glVertex3f(x, 0, GridMax.y());
	}
	#pragma endregion
	glEnd();
}
void OpenGLWidget::DrawPointCloud()
{
	if (rawManager != NULL && rawManager->PointCloudArray.size() > 0)
	{
		float pointSize = (1 - (float)(Radius - MinRadius) / (MaxRadius - MinRadius)) * 0.1f;
		glPointSize(pointSize);
		glBegin(GL_POINTS);
		for (int i = 0; i < rawManager->PointCloudArray.size(); i++)
		{
			glColor4f(0, 0, 0, 1);
			for (int j = 0; j < rawManager->PointCloudArray[i].Points.size(); j++)
			{
				QVector3D point = rawManager->PointCloudArray[i].Points[j];
				glVertex3f(point.x(), point.y(), point.z());
			}
		}
		glEnd();
	}
}
void OpenGLWidget::DrawSTL()
{
	#pragma region 先判斷是不是空的
	if (!IsLoaded)
		return;
	#pragma endregion
	#pragma region 跑每一個點，把它畫出來
	// 設定 Uniform Data
	program->bind();
	program->setUniformValue(ProjectionMatrixLoc,	ProjectionMatrix);
	program->setUniformValue(ViewMatrixLoc,			ViewMatrix);
	program->setUniformValue(ModelMatrixLoc,		TransformMatrix);

	// 傳資料上去
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, BaryCentricBuffer);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glDrawArrays(GL_TRIANGLES, 0, VertexData.size());
	program->release();
	//if (RenderTriangle_bool)
	//{
	//	glColor4f(0.968f, 0.863f, 0.445f, 1);					// #F7DD72
	//	glBegin(GL_TRIANGLES);
	//	for (MeshType::FaceIter f_iter = STLFile.faces_begin(); f_iter != STLFile.faces_end(); f_iter++)
	//		for (MeshType::FaceVertexIter fv_iter = STLFile.fv_iter(f_iter); fv_iter.is_valid(); fv_iter++)
	//		{
	//			MeshType::Point p = STLFile.point(fv_iter);

	//			// 算出矩陣結果
	//			QVector4D matrixP(p[0], p[1], p[2], 1);
	//			matrixP = TransformMatrix * matrixP;

	//			// 畫出來
	//			glVertex3f(
	//				matrixP[0] + OffsetToCenter.x(),
	//				matrixP[1] + OffsetToCenter.y(),
	//				matrixP[2] + OffsetToCenter.z()
	//			);
	//		}
	//	glEnd();
	//}
	//if (RenderBorder_bool)
	//{
	//	glColor4f(0, 0, 0, 1);
	//	glBegin(GL_LINES);
	//	for (MeshType::FaceIter f_iter = STLFile.faces_begin(); f_iter != STLFile.faces_end(); f_iter++)
	//		for (MeshType::FaceEdgeIter fe_iter = STLFile.fe_iter(f_iter); fe_iter.is_valid(); fe_iter++)
	//		{

	//			MeshType::Point FirstP = STLFile.point(STLFile.to_vertex_handle(STLFile.halfedge_handle(fe_iter, 0)));
	//			MeshType::Point SecondP = STLFile.point(STLFile.to_vertex_handle(STLFile.halfedge_handle(fe_iter, 1)));

	//			// 算出矩陣結果
	//			QVector4D matrixFirstP(FirstP[0], FirstP[1], FirstP[2], 1);
	//			QVector4D matrixSecondP(SecondP[0], SecondP[1], SecondP[2], 1);
	//			matrixFirstP = TransformMatrix * matrixFirstP;
	//			matrixSecondP = TransformMatrix * matrixSecondP;

	//			glVertex3f(
	//				matrixFirstP[0] + OffsetToCenter.x(),
	//				matrixFirstP[1] + OffsetToCenter.y(),
	//				matrixFirstP[2] + OffsetToCenter.z()
	//			);
	//			glVertex3f(
	//				matrixSecondP[0] + OffsetToCenter.x(),
	//				matrixSecondP[1] + OffsetToCenter.y(),
	//				matrixSecondP[2] + OffsetToCenter.z()
	//			);
	//		}
	//	glEnd();
	//}

	//// 點雲
	//if (RenderPointCloud_bool)
	//{
	//	glColor4f(0, 0, 0, 1);
	//	glPointSize(3);
	//	glBegin(GL_POINTS);
	//	for (int i = 0; i < PointArray.size(); i++)
	//	{
	//		QVector3D p = PointArray[i];
	//		QVector4D vec4P = QVector4D(p, 1);
	//		vec4P = TransformMatrix * vec4P;
	//		glVertex3f(vec4P.x() + OffsetToCenter.x(), vec4P.y() + OffsetToCenter.y(), vec4P.z() + OffsetToCenter.z());
	//	}
	//	glEnd();
	//}
	#pragma endregion
}
void OpenGLWidget::DrawResetRotation()
{
	glPushMatrix();
	glLoadIdentity();
	glTranslatef(0, 2, 0);

	glLineWidth(10);

	// 旋轉
	QVector3D dir = rawManager->bleManager.GetAngle();
	//cout << dir.x() << " " << dir.y() << " " << dir.z() << endl;
	glRotatef(dir.x(), 1, 0, 0);
	glRotatef(dir.y(), 0, 1, 0);
	glRotatef(dir.z(), 0, 0, 1);

	// X 
	glColor4f(1, 0, 0, 1);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(5, 0, 0);
	glEnd();

	// Y
	glColor4f(1, 1, 0, 1);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 5, 0);
	glEnd();

	// Z 
	glColor4f(0, 1, 0, 1);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, 5);
	glEnd();

	glLineWidth(1);
	glPopMatrix();
}
