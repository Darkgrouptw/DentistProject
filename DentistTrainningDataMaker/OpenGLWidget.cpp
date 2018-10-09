#include "OpenGLWidget.h"

OpenGLWidget::OpenGLWidget(QWidget* parent = 0) : QOpenGLWidget(parent)
{
}
OpenGLWidget::~OpenGLWidget()
{
}

void OpenGLWidget::initializeGL()
{
	initializeOpenGLFunctions();
	glClearColor(0.5f, 0.5, 0.5f, 1);
	glEnable(GL_DEPTH_TEST);

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
	DrawSTL();
}

// 滑鼠事件
void OpenGLWidget::mousePressEvent(QMouseEvent *event)
{
	PressPoint = event->pos();
	TempArcAngle = ArcAngle;
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
	#pragma region 讀取 STL 檔案
	if (!OpenMesh::IO::read_mesh(STLFile, FileName.toStdString()))
	{
		cout << "讀取錯誤!!" << endl;
		IsLoaded = false;
		return false;
	}
	cout << "模型面數：" << STLFile.n_faces() << endl;
	IsLoaded = true;
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
	cout << "MaxDelta " << maxDelta << endl;

	float scale = GridSize / maxDelta * 2;					// 因為是 -10 ~ 10，所以是 20
	TransformMatrix.scale(QVector3D(scale, scale, scale));
	cout << "Scale " << scale << endl;

	// 這邊要找 offset 多少
	QVector4D downPos;
	bool TakeY = true;
	if (deltaY > deltaX || deltaY > deltaZ)
	{
		downPos = QVector4D(0, minZ, 0, 1);
		TakeY = false;
	}
	else
		downPos = QVector4D(0, minY, 0, 1);

	downPos = TransformMatrix * downPos;
	cout << downPos[0] << " " << downPos[1] << " " << downPos[2] << " " << downPos[3] << endl;

	if (!TakeY)
		OffsetY = downPos.z();
	else
		OffsetY = -downPos.y();

	cout << "OffsetY " << OffsetY << endl;
	#pragma endregion
	return true;
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
void OpenGLWidget::DrawSTL()
{
	#pragma region 先判斷是不是空的
	if (!IsLoaded)
		return;
	#pragma endregion
	#pragma region 跑每一個點，把它畫出來 (暫時先用 OpenGL 1 來畫)
	glColor4f(0.968f, 0.863f, 0.445f, 1);					// #F7DD72
	glBegin(GL_TRIANGLES);
	for (MeshType::FaceIter f_iter = STLFile.faces_begin(); f_iter != STLFile.faces_end(); f_iter++)
		for (MeshType::FaceVertexIter fv_iter = STLFile.fv_iter(f_iter); fv_iter.is_valid(); fv_iter++)
		{
			MeshType::Point p = STLFile.point(fv_iter);

			// 算出矩陣結果
			QVector4D matrixP(p[0], p[1], p[2], 1);
			matrixP = TransformMatrix * matrixP;

			// 畫出來
			glVertex3f(matrixP[0], matrixP[1] + OffsetY, matrixP[2]);
		}
	glEnd();

	glColor4f(0, 0, 0, 1);
	glBegin(GL_LINES);
	for (MeshType::FaceIter f_iter = STLFile.faces_begin(); f_iter != STLFile.faces_end(); f_iter++)
		for (MeshType::FaceEdgeIter fe_iter = STLFile.fe_iter(f_iter); fe_iter.is_valid(); fe_iter++)
		{
			
			MeshType::Point FirstP = STLFile.point(STLFile.to_vertex_handle(STLFile.halfedge_handle(fe_iter, 0)));
			MeshType::Point SecondP = STLFile.point(STLFile.to_vertex_handle(STLFile.halfedge_handle(fe_iter, 1)));

			// 算出矩陣結果
			QVector4D matrixFirstP(FirstP[0], FirstP[1], FirstP[2], 1);
			QVector4D matrixSecondP(SecondP[0], SecondP[1], SecondP[2], 1);
			matrixFirstP = TransformMatrix * matrixFirstP;
			matrixSecondP = TransformMatrix * matrixSecondP;

			glVertex3f(matrixFirstP[0], matrixFirstP[1] + OffsetY, matrixFirstP[2]);
			glVertex3f(matrixSecondP[0], matrixSecondP[1] + OffsetY, matrixSecondP[2]);
		}
	glEnd();
	#pragma endregion
}
