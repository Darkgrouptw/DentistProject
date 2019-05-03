#include "Display_BoundingBox.h"

Display_BoundingBox::Display_BoundingBox(QWidget* parent = NULL) : QOpenGLWidget(parent)
{
#pragma region 初始化貼圖
	texture = new GLuint[1];
#pragma endregion
}
Display_BoundingBox::~Display_BoundingBox()
{
#pragma region 刪除貼圖
	delete[] texture;
#pragma endregion
}

// 繪製 Function
void Display_BoundingBox::initializeGL()
{
	initializeOpenGLFunctions();
	glClearColor(0.5, 0.5, 0.7, 1);

	CalcMatrix();
	SetFbo();

	square = new Square();
	square->Init("./Shaders/Square.vs", "./Shaders/Square.fs");
}
void Display_BoundingBox::paintGL()
{
//#pragma region FBO測試
//	glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glViewport(0, 0, 500, 500);
//
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//
//	glPushMatrix();
//	glBegin(GL_QUADS);
//	glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, 0.0f);
//	glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(1.0f, -1.0f, 0.0f);
//	glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(1.0f, 1.0f, 0.0f);
//	glColor3f(1.0f, 1.0f, 1.0f); glVertex3f(-1.0f, 1.0f, 0.0f);
//	glEnd();
//	glPopMatrix();
//
//	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
//	//glDrawBuffer(GL_BACK);
//	glClearColor(0.5, 0.5, 0.7, 1);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
//	glViewport(0, 0, 500, 500);
//	
//
//	
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	//glLoadMatrixf((ProjectionMatrix * ViewMatrix).data());
//
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//
//	glGetFloatv(GL_MODELVIEW_MATRIX, ModelViewMatrixS);
//	glGetFloatv(GL_PROJECTION_MATRIX, ProjectionMatrixS);
//	
//	//glEnable(GL_TEXTURE_2D);
//	square->Begin();
//	//glActiveTexture(GL_TEXTURE1);
//	//glBindTexture(GL_TEXTURE_2D, _fbo_C);
//	//square->shaderProgram->setUniformValue("Texture", 1);
//	square->Paint(ProjectionMatrixS, ModelViewMatrixS);
//	//glActiveTexture(GL_TEXTURE0);
//	//glDisable(GL_TEXTURE_2D);
//#pragma endregion

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glLoadMatrixf((ProjectionMatrix * ViewMatrix).data());

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glScalef(scaleSize, scaleSize, 1.0);
	glTranslatef(movex, movey, 0.0);

	glGetIntegerv(GL_VIEWPORT, view);

	//glShadeModel(GL_FLAT);
	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	//glColor3f(0.5, 0.5, 0);
	glColor3f(1.0, 1.0, 1.0);
	glBindTexture(GL_TEXTURE_2D, texture[0]);

	glPushMatrix();
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-InitPainterSize / 2 * WHProportion, InitPainterSize / 2);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-InitPainterSize / 2 * WHProportion, -InitPainterSize / 2);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(InitPainterSize / 2 * WHProportion, -InitPainterSize / 2);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(InitPainterSize / 2 * WHProportion, InitPainterSize / 2);
	glEnd();
	glDisable(GL_TEXTURE_2D);
	glPopMatrix();

	if (TempAreaPoint.size() > 0) {
		glPushMatrix();
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);

		for (int i = 0; i < TempAreaPoint.size() - 1; i++) {
			glVertex2f(TempAreaPoint[i%TempAreaPoint.size()].x(), TempAreaPoint[i%TempAreaPoint.size()].y());
			glVertex2f(TempAreaPoint[(i + 1) % TempAreaPoint.size()].x(), TempAreaPoint[(i + 1) % TempAreaPoint.size()].y());
		}

		glEnd();
		glPopMatrix();
	}
	
}

// 滑鼠事件
void Display_BoundingBox::mousePressEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton) {
		DrawTranslateChange = false;
		TempAreaPoint.clear();
		float tempX = ((float)(event->pos().x()) / view[2] * 2.0f - 1.0f) / scaleSize - movex;
		float tempY = ((float)(view[3] - event->pos().y()) / view[3] * 2.0f - 1.0f) / scaleSize - movey;
		
		////判斷有無超出畫布
		if (tempX < -InitPainterSize / 2 * WHProportion) tempX = -InitPainterSize / 2 * WHProportion;
		else if (tempX > InitPainterSize / 2 * WHProportion)tempX = InitPainterSize / 2 * WHProportion;
		if (tempY < -InitPainterSize / 2)tempY = -InitPainterSize / 2;
		else if (tempY > InitPainterSize / 2)tempY = InitPainterSize / 2;

		TempAreaPoint.push_back(QVector2D(tempX, tempY));
	}
	else if (event->button() == Qt::RightButton) {
		DrawTranslateChange = true;
		nowx = event->pos().x();
		nowy = event->pos().y();
		nextx = event->pos().x();
		nexty = event->pos().y();
	}
}
void Display_BoundingBox::mouseMoveEvent(QMouseEvent *event)
{
	if (DrawTranslateChange) {
		nextx = event->pos().x();
		nexty = event->pos().y();
		if (nowx != nextx)movex += (nextx - nowx) / 800.0f;
		if (nowy != nexty)movey -= (nexty - nowy) / 800.0f;
		nowx = event->pos().x();
		nowy = event->pos().y();
	}
	else {
		float tempX = ((float)(event->pos().x()) / view[2] * 2.0f - 1.0f) / scaleSize - movex;
		float tempY = ((float)(view[3] - event->pos().y()) / view[3] * 2.0f - 1.0f) / scaleSize - movey;
		
		////判斷有無超出畫布
		if (tempX < -InitPainterSize / 2 * WHProportion) tempX = -InitPainterSize / 2 * WHProportion;
		else if (tempX > InitPainterSize / 2 * WHProportion)tempX = InitPainterSize / 2 * WHProportion;
		if (tempY < -InitPainterSize / 2)tempY = -InitPainterSize / 2;
		else if (tempY > InitPainterSize / 2)tempY = InitPainterSize / 2;

		TempAreaPoint.push_back(QVector2D(tempX, tempY));
	}
	this->update();
}
void Display_BoundingBox::mouseReleaseEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton)
		TempAreaPoint.push_back(TempAreaPoint[0]);
	this->update();
}
void Display_BoundingBox::wheelEvent(QWheelEvent *event)
{
	if ((scaleSize + (float)event->angleDelta().y() / 800.0f) > 0)
		scaleSize += (float)event->angleDelta().y() / 800.0f;

	// 更新 Widget
	CalcMatrix();
	this->update();
}

// 外部連結
void Display_BoundingBox::LoadTexture(QImage img, int index)
{
	QImage t = QGLWidget::convertToGLFormat(img);

	TexWidth = t.width();
	TexHeight = t.height();
	WHProportion = (float)t.width() / t.height();

	glGenTextures(1, &texture[index]);
	glBindTexture(GL_TEXTURE_2D, texture[index]);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, t.width(), t.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, t.bits());

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}

// 矩陣相關
void Display_BoundingBox::CalcMatrix()
{
	ProjectionMatrix.setToIdentity();
	ProjectionMatrix.ortho(-1, 1, -1, 1, -10, 10);

	ViewMatrix.setToIdentity();
	ViewMatrix.lookAt(
		QVector3D(0, 0, 5),
		QVector3D(0, 0, 0),
		QVector3D(0, 1, 0)
	);
}

// FboInit
void Display_BoundingBox::SetFbo()
{
	glGenFramebuffers(1, &_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, _fbo);

	glGenTextures(1, &_fbo_C);
	glBindTexture(GL_TEXTURE_2D, _fbo_C);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, this->width(), this->height(), 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _fbo_C, 0);

	glGenRenderbuffers(1, &_fbo_D);
	glBindRenderbuffer(GL_RENDERBUFFER, _fbo_D);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, this->width(), this->height());
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _fbo_D);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}