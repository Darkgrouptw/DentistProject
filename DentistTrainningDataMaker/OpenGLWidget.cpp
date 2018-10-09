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
void OpenGLWidget::mousePressEvent(QMouseEvent *e)
{
	PressPoint = e->pos();
	TempArcAngle = ArcAngle;
}
void OpenGLWidget::mouseMoveEvent(QMouseEvent *e)
{
	CurrentPoint = e->pos();

	int width = CurrentPoint.x() - PressPoint.x();
	float rate = (float)width / this->width();

	ArcAngle = TempArcAngle + rate * 180;
	cout << ArcAngle << endl;
	CalcMatrix();

	this->update();
}

//////////////////////////////////////////////////////////////////////////
// 外部呼叫函數
//////////////////////////////////////////////////////////////////////////
bool OpenGLWidget::LoadSTLFile(QString FileName)
{
	if (!OpenMesh::IO::read_mesh(STLFile, FileName.toStdString()))
	{
		cout << "讀取錯誤!!" << endl;
		IsLoaded = false;
		return false;
	}
	cout << "模型面數：" << STLFile.n_faces() << endl;
	IsLoaded = true;
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
			glVertex3f(p[0], p[1], p[2]);
		}
	glEnd();

	glColor4f(0, 0, 0, 1);
	glBegin(GL_LINES);
	for (MeshType::FaceIter f_iter = STLFile.faces_begin(); f_iter != STLFile.faces_end(); f_iter++)
		for (MeshType::FaceEdgeIter fe_iter = STLFile.fe_iter(f_iter); fe_iter.is_valid(); fe_iter++)
		{
			
			MeshType::Point FirstP = STLFile.point(STLFile.to_vertex_handle(STLFile.halfedge_handle(fe_iter, 0)));
			MeshType::Point SecondP = STLFile.point(STLFile.to_vertex_handle(STLFile.halfedge_handle(fe_iter, 1)));
			glVertex3f(FirstP[0], FirstP[1], FirstP[2]);
			glVertex3f(SecondP[0], SecondP[1], SecondP[2]);
		}
	glEnd();
	#pragma endregion
}
