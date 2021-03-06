﻿#include "OpenGLWidget.h"

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
	#pragma region 設定 Rotation 的矩陣
	OCTView_ModelMatrix[0].setToIdentity();
	OCTView_ModelMatrix[0].rotate(90, 1, 0, 0);
	OCTView_ModelMatrix[0].translate(0, -5, -5);
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
	{
		DrawPointCloud();
		//DrawVolumeData();
	}
	//DrawVolumeData();

	#ifdef DEBUG_DRAW_AVERAGE_ERROR_PC
	// Debug 用
	DrawAverageDebugPC();
	#endif
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
void OpenGLWidget::SetRotationMode(bool SetBool)
{
	RotationMode = SetBool;
	this->update();
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
	#pragma region VolumeData
	tempInfo = LinkProgram("./Shaders/VolumeData");
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
	if (rawManager != NULL && rawManager->PointCloudArray.size() > 0)
	{
		// 這邊是要先判斷有沒有 Lock
		// 如果有 Lock 代表說，要更新
		// 但由於 OpenGL 只能有單一一個 Thread 去呼叫
		// 其他 Thread 去呼叫此 Class 的 Function 都會出現 1282 (GL_INVALID_OPERATION)
		if (rawManager->IsLockPC)
		{
			UpdatePC();
			rawManager->IsLockPC = false;
		}

		assert(ProgramList.size() >= 3);

		QOpenGLShaderProgram* program = ProgramList[2].program;
		program->bind();

		program->setUniformValue(ProgramList[2].ProjectionMLoc, ProjectionMatrix);
		program->setUniformValue(ProgramList[2].ViewMLoc,		ViewMatrix);
		program->setUniformValue(ProgramList[2].ModelMLoc,		OCTView_ModelMatrix[OCTViewType]);

		float pointSize = (1 - (float)(Radius - MinRadius) / (MaxRadius - MinRadius)) * 0.1f;
		program->setUniformValue(PointSizeLoc, pointSize);

		// 畫
		for (int i = 0; i < PointCloudVertexBufferList.size(); i++)
		{
			//int i = PointCloudVertexBufferList.size() - 1;
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

	program->setUniformValue(ProgramList[1].ProjectionMLoc, ProjectionMatrix);
	program->setUniformValue(ProgramList[1].ViewMLoc,		ViewMatrix);
	program->setUniformValue(ProgramList[1].ModelMLoc,		GyroTranslateM);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, GyroModelVertexBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glDrawArrays(GL_TRIANGLES, 0, GyroModelPoints.size());

	program->release();
}
#ifdef DEBUG_DRAW_AVERAGE_ERROR_PC
void OpenGLWidget::DrawAverageDebugPC()
{
	#pragma region 坐標軸測試
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(OCTView_ModelMatrix->data());
	glLineWidth(3);

	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_LINES);
		glVertex3f(1.0, 0.0, 0.0);
		glVertex3f(0, 0, 0);
	glEnd();

	glColor3f(1.0, 1.0, 0.0);
	glBegin(GL_LINES);
		glVertex3f(0.0, 1.0, 0.0);
		glVertex3f(0, 0, 0);
	glEnd();

	glColor3f(0.0, 0.0, 1.0);
	glBegin(GL_LINES);
	glVertex3f(0.0, 0.0, 1.0);
	glVertex3f(0, 0, 0);
	glEnd();

	glPopMatrix();
	#pragma endregion

	// 先判斷是否有判斷過
	if (rawManager->AllCenterPoint.x() == 0 && rawManager->AllCenterPoint.y() == 0 && rawManager->AllCenterPoint.z() == 0)
		return;

	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(OCTView_ModelMatrix->data());

	// 線
	for (int i = 0; i < rawManager->PointCloudArray.size(); i++) 
	{
		glBegin(GL_LINES);
		glColor3f(1.0, 0.0, 0.0);
		glVertex3f(rawManager->AllCenterPoint.x(), rawManager->AllCenterPoint.y(), rawManager->AllCenterPoint.z());
		glVertex3f(rawManager->PointCloudArray[i].CenterPoint.x(), rawManager->PointCloudArray[i].CenterPoint.y(), rawManager->PointCloudArray[i].CenterPoint.z());
		glEnd();
	}

	glBegin(GL_QUADS);
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(rawManager->PlanePoint[0].x(), rawManager->PlanePoint[0].y(), rawManager->PlanePoint[0].z());
	glVertex3f(rawManager->PlanePoint[1].x(), rawManager->PlanePoint[1].y(), rawManager->PlanePoint[1].z());
	glVertex3f(rawManager->PlanePoint[2].x(), rawManager->PlanePoint[2].y(), rawManager->PlanePoint[2].z());
	glVertex3f(rawManager->PlanePoint[3].x(), rawManager->PlanePoint[3].y(), rawManager->PlanePoint[3].z());
	glEnd();

	// 畫 Upper Vecot
	if (rawManager->UpperVector != QVector3D())
	{
		glColor3f(1.0, 1.0, 0.0);
		glBegin(GL_LINES);
		glVertex3f(rawManager->AllCenterPoint.x(), rawManager->AllCenterPoint.y(), rawManager->AllCenterPoint.z());
		glVertex3f(rawManager->UpperVector.x(), rawManager->UpperVector.y(), rawManager->UpperVector.z());
		glEnd();
	}

	glPopMatrix();
}
#endif

//void OpenGLWidget::DrawVolumeData()
//{
//	if (rawManager != NULL && rawManager->VolumeDataArray.size() > 0)
//	{
//		// 這邊是要先判斷有沒有 Lock
//		// 如果有 Lock 代表說，要更新
//		// 但由於 OpenGL 只能有單一一個 Thread 去呼叫
//		// 其他 Thread 去呼叫此 Class 的 Function 都會出現 1282 (GL_INVALID_OPERATION)
//		if (rawManager->IsLockVolumeData)
//		{
//			UpdateVolumeData();
//			rawManager->IsLockVolumeData = false;
//		}
//
//		assert(ProgramList.size() >= 4);
//
//		QOpenGLShaderProgram* program = ProgramList[3].program;
//		program->bind();
//
//		QMatrix4x4 modelM;
//		modelM.setToIdentity();
//		program->setUniformValue(ProgramList[3].ProjectionMLoc, ProjectionMatrix);
//		program->setUniformValue(ProgramList[3].ViewMLoc, ViewMatrix);
//		program->setUniformValue(ProgramList[3].ModelMLoc, modelM);
//
//		// 畫
//		for (int i = 0; i < VolumeDataVertexBufferList.size(); i++)
//		{
//			glBindBuffer(GL_ARRAY_BUFFER, VolumeDataVertexBufferList[i]);
//			glEnableVertexAttribArray(0);
//			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
//
//			glBindBuffer(GL_ARRAY_BUFFER, VolumeDataPointTypeBufferList[i]);
//			glEnableVertexAttribArray(1);
//			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, NULL);
//
//
//			int PointSize = rawManager->VolumeDataArray[i]->Points.size();
//			glDrawArrays(GL_TRIANGLES, 0, PointSize);
//		}
//		program->release();
//	}
//}

// 更新 Buffer
void OpenGLWidget::UpdatePC()
{
	// Program 設定
	assert(ProgramList.size() >= 3);

	QOpenGLShaderProgram* program = ProgramList[2].program;
	program->bind();

	// 刪除 Buffer
	for (int i = 0; i < PointCloudVertexBufferList.size(); i++)
		glDeleteBuffers(1, &PointCloudVertexBufferList[i]);
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
//void OpenGLWidget::UpdateVolumeData()
//{
//	// Program 設定
//	assert(ProgramList.size() >= 4);
//
//	QOpenGLShaderProgram* program = ProgramList[3].program;
//	program->bind();
//
//	// 刪除 Buffer
//	for (int i = 0; i < VolumeDataVertexBufferList.size(); i++)
//	{
//		glDeleteBuffers(1, &VolumeDataVertexBufferList[i]);
//		glDeleteBuffers(1, &VolumeDataPointTypeBufferList[i]);
//	}
//	VolumeDataVertexBufferList.clear();
//	VolumeDataPointTypeBufferList.clear();
//
//	for (int i = 0; i < rawManager->VolumeDataArray.size(); i++)
//	{
//		// 拿資料
//		QVector<QVector3D> pointData;
//		rawManager->VolumeDataArray[i]->GenerateMeshFromLookZ();
//		QVector<QVector3D>& tempP = rawManager->VolumeDataArray[i]->Points;
//		QVector<float>& tempT = rawManager->VolumeDataArray[i]->PointType;
//
//		GLuint VBuffer;
//		glGenBuffers(1, &VBuffer);
//		glBindBuffer(GL_ARRAY_BUFFER, VBuffer);
//		glBufferData(GL_ARRAY_BUFFER, sizeof(QVector3D) * tempP.size(), tempP.constData(), GL_STATIC_DRAW);
//
//		VolumeDataVertexBufferList.push_back(VBuffer);
//
//		GLuint TypeBuffer;
//		glGenBuffers(1, &TypeBuffer);
//		glBindBuffer(GL_ARRAY_BUFFER, TypeBuffer);
//		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * tempT.size(), tempT.constData(), GL_STATIC_DRAW);
//
//		VolumeDataPointTypeBufferList.push_back(TypeBuffer);
//	}
//	program->release();
//}