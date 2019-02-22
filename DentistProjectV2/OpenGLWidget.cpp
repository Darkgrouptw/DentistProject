#include "OpenGLWidget.h"

OpenGLWidget::OpenGLWidget(QWidget* parent = 0) : QOpenGLWidget(parent)
{
	#pragma region 讀模型
	QVector<QVector2D>	GyroModelUVs;
	QVector<QVector3D>	GyroModelNormals;
	QVector<unsigned int> MaterialIndex;
	QVector<QString> MaterialName;
	OBJLoader::loadOBJ("./Models/handpiece.obj", GyroModelPoints, GyroModelUVs, GyroModelNormals, MaterialIndex, MaterialName);
	cout << "讀取九軸矯正模型!!" << endl;
	#pragma endregion

}
OpenGLWidget::~OpenGLWidget()
{
	// 刪除 Program
	for (int i = 0; i < ProgramList.count(); i++)
		delete ProgramList[i].program;
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
		DrawPointCloud();
}

// 滑鼠事件
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

// 外部呼叫函數
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
void OpenGLWidget::UpdatePC()
{
	// Prgram 設定
	assert(ProgramList.size() >= 3);

	QOpenGLShaderProgram* program = ProgramList[2].program;
	program->bind();

	// 刪除 Buffer
	for (int i = 0; i < PointCloudVertexBufferList.size(); i++)
	{
		glBindBuffer(GL_ARRAY_BUFFER, PointCloudVertexBufferList[i]);
		glDeleteBuffers(1, &PointCloudVertexBufferList[i]);
	}
	PointCloudVertexBufferList.clear();

	// 加資料
	for (int i = 0; i < rawManager->PointCloudArray.size(); i++)
	{
		QVector<QVector3D> &tempPC = rawManager->PointCloudArray[i].Points;

		GLuint VBuffer;
		glGenBuffers(1, &VBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, VBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(QVector3D) * tempPC.size(), tempPC.constData(), GL_STATIC_DRAW);

		PointCloudVertexBufferList.push_back(VBuffer);
	}

	program->release();
}

// 其他元件的 Function
void OpenGLWidget::SetRawDataManager(RawDataManager* raw)
{
	rawManager = raw;
}

// 初始化
void OpenGLWidget::InitProgram()
{
	#pragma region Ground
	ProgramInfo tempInfo = LinkProgram("./Shaders/DrawGound");

	// 點 & UV
	GroundPoints.push_back(QVector3D(GridMin.x(), 0, GridMin.y()));
	GroundPoints.push_back(QVector3D(GridMax.x(), 0, GridMin.y()));
	GroundPoints.push_back(QVector3D(GridMin.x(), 0, GridMax.y()));
	GroundPoints.push_back(QVector3D(GridMax.x(), 0, GridMax.y()));

	GroundUVs.push_back(QVector2D(0, 0));
	GroundUVs.push_back(QVector2D(1, 0));
	GroundUVs.push_back(QVector2D(0, 1));
	GroundUVs.push_back(QVector2D(1, 1));

	float lowerY = -0.01f;
	GroundModelM.setToIdentity();
	GroundModelM.translate(0, lowerY, 0);

	// 上傳
	tempInfo.program->bind();

	glGenBuffers(1, &GroundVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, GroundVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, GroundPoints.size() * sizeof(QVector3D), GroundPoints.constData(), GL_STATIC_DRAW);

	glGenBuffers(1, &GroundUVBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, GroundUVBuffer);
	glBufferData(GL_ARRAY_BUFFER, GroundUVs.size() * sizeof(QVector2D), GroundUVs.constData(), GL_STATIC_DRAW);

	tempInfo.program->release();

	ProgramList.push_back(tempInfo);
	#pragma endregion
	#pragma region Model
	tempInfo = LinkProgram("./Shaders/Model");
	GyroTranslateM.setToIdentity();
	GyroTranslateM.translate(0, 5, 0);
	GyroTranslateM.scale(1.2f);

	// 上傳
	tempInfo.program->bind();

	glGenBuffers(1, &GyroModelVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, GyroModelVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, GyroModelPoints.size() * sizeof(QVector3D), GyroModelPoints.constData(), GL_STATIC_DRAW);

	tempInfo.program->release();

	ProgramList.push_back(tempInfo);
	#pragma endregion
	#pragma region Point
	tempInfo = LinkProgram("./Shaders/PointCloud");

	tempInfo.program->bind();

	// Location
	PointSizeLoc        = tempInfo.program->uniformLocation("pointSize");
	IsCrurrentPCLoc     = tempInfo.program->uniformLocation("IsCurrentPC");

	tempInfo.program->release();

	ProgramList.push_back(tempInfo);
	#pragma endregion
}
ProgramInfo OpenGLWidget::LinkProgram(QString path)
{
	ProgramInfo tempInfo;
	tempInfo.program = new QOpenGLShaderProgram();
	tempInfo.program->addShaderFromSourceFile(QOpenGLShader::Vertex,	path + ".vert");
	tempInfo.program->addShaderFromSourceFile(QOpenGLShader::Fragment,	path + ".frag");
	tempInfo.program->link();

	tempInfo.ProjectionMLoc	= tempInfo.program->uniformLocation("ProjectionMatrix");
	tempInfo.ViewMLoc		= tempInfo.program->uniformLocation("ViewMatrix");
	tempInfo.ModelMLoc		= tempInfo.program->uniformLocation("ModelMatrix");
	return tempInfo;
}

// 矩陣相關
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

// 畫畫的 Function
void OpenGLWidget::DrawGround()
{
	assert(ProgramList.size() >= 1);

	QOpenGLShaderProgram* program = ProgramList[0].program;
	program->bind();

	program->setUniformValue(ProgramList[0].ProjectionMLoc, ProjectionMatrix);
	program->setUniformValue(ProgramList[0].ViewMLoc,		ViewMatrix);
	program->setUniformValue(ProgramList[0].ModelMLoc,		GroundModelM);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, GroundVertexBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, GroundUVBuffer);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, GroundPoints.size());
	program->release();
}
void OpenGLWidget::DrawPointCloud()
{
	if (rawManager != NULL && rawManager->PointCloudArray.size() > 0 && !rawManager->IsLockPC)
	{
		assert(ProgramList.size() >= 3);

		QOpenGLShaderProgram* program = ProgramList[2].program;
		program->bind();

		QMatrix4x4 modelM;
		modelM.setToIdentity();
		program->setUniformValue(ProgramList[2].ProjectionMLoc, ProjectionMatrix);
		program->setUniformValue(ProgramList[2].ViewMLoc,		ViewMatrix);

		float pointSize = (1 - (float)(Radius - MinRadius) / (MaxRadius - MinRadius)) * 0.1f;
		program->setUniformValue(PointSizeLoc, pointSize);

		// 畫
		for (int i = 0; i < rawManager->PointCloudArray.size(); i++)
		{
			if (rawManager->SelectIndex == i)
				program->setUniformValue(IsCrurrentPCLoc, true);
			else
				program->setUniformValue(IsCrurrentPCLoc, false);

			glBindBuffer(GL_ARRAY_BUFFER, PointCloudVertexBufferList[i]);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
			
			glDrawArrays(GL_POINTS, 0, rawManager->PointCloudArray[i].Points.size());
		}
		program->release();
	}
}
void OpenGLWidget::DrawResetRotation()
{
	assert(ProgramList.size() >= 2);
	QOpenGLShaderProgram* program = ProgramList[1].program;
	program->bind();

	QMatrix4x4 rotationM;
	rotationM.setToIdentity();
	rotationM.rotate(rawManager->bleManager.GetQuaternionFromDevice());

	program->setUniformValue(ProgramList[1].ProjectionMLoc, ProjectionMatrix);
	program->setUniformValue(ProgramList[1].ViewMLoc,		ViewMatrix);
	program->setUniformValue(ProgramList[1].ModelMLoc,		GyroTranslateM * rotationM);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, GyroModelVertexBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glDrawArrays(GL_TRIANGLES, 0, GyroModelPoints.size());

	program->release();
}
