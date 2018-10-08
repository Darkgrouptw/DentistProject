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
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glLoadMatrixf((ProjectionMatrix * ViewMatrix).data());

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void OpenGLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	DrawGrid();

}

// 外部呼叫函數
bool OpenGLWidget::LoadSTLFile(QString FileName)
{
	if (!OpenMesh::IO::read_mesh(STLFile, FileName.toStdString()))
	{
		cout << "讀取錯誤!!" << endl;
		return false;
	}
	cout << "模型點數：" << STLFile.n_faces() << endl;
	return true;
}

// 矩陣相關
void OpenGLWidget::CalcMatrix()
{
	float AngleInRadian = ElevationAngle * M_PI / 180;
	float GoundRadius = Radius * cos(AngleInRadian);
	float Height = Radius * sin(AngleInRadian);

	float ArcAngleInRadian = ArcAngle * M_PI / 180;
	float GoundX = GoundRadius * cos(ArcAngleInRadian);
	float GoundZ = GoundRadius * sin(ArcAngleInRadian);

	ProjectionMatrix.setToIdentity();
	ProjectionMatrix.perspective(60, 1, 0.1f, 100);

	ViewMatrix.setToIdentity();
	ViewMatrix.lookAt(
		QVector3D(GoundX, Height, GoundZ),
		QVector3D(0, 0, 0),
		QVector3D(0, 1, 0)
	);
}

void OpenGLWidget::DrawGrid()
{
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
