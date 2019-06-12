#pragma once
#include <QOpenGLWidget>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>

class DisplayPanel : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
	DisplayPanel(QWidget *);
	~DisplayPanel();

	//////////////////////////////////////////////////////////////////////////
	// 繪畫 Function
	//////////////////////////////////////////////////////////////////////////
};

