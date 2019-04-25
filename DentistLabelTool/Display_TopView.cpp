#include "Display_TopView.h"

Display_TopView::Display_TopView(QWidget* parent=NULL) : QOpenGLWidget(parent)
{
	#pragma region 初始化貼圖
	texture = new GLuint[1];
	#pragma endregion
}
Display_TopView::~Display_TopView()
{
	#pragma region 刪除貼圖
	delete[] texture;
	#pragma endregion
}

// 繪製 Function
void Display_TopView::initializeGL()
{
	initializeOpenGLFunctions();
	glClearColor(0.5, 0.5, 0.7, 1);
}
void Display_TopView::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();

	//glShadeModel(GL_FLAT);
	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	//glColor3f(0.5, 0.5, 0);
	glBindTexture(GL_TEXTURE_2D, texture[0]);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-0.5f, 0.5f);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-0.5f, -0.5f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(0.5f, -0.5f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(0.5f, 0.5f);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

// 外部連結
void Display_TopView::LoadTexture(QImage img, int index)
{
	QImage t = QGLWidget::convertToGLFormat(img);

	glGenTextures(1, &texture[index]);
	glBindTexture(GL_TEXTURE_2D, texture[index]);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, t.width(), t.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, t.bits());

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}