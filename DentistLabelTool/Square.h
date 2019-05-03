#pragma once
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLShader>
#include <QtGui/QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QVector>
#include <QVector3D>
#include <QFileInfo>
#include <QDebug>
#include <QString>
#include <QImage>

class Square : QOpenGLFunctions_4_3_Core
{
public:
	QOpenGLShaderProgram* shaderProgram;
	QOpenGLShader* vertexShader;
	QOpenGLShader* fragmentShader;
	QVector<QVector3D> vertices;
	QVector<QVector2D> uvs;
	QOpenGLVertexArrayObject vao;
	QOpenGLBuffer vvbo;
	QOpenGLBuffer uvbo;
public:
	Square();
	~Square();
	void Init(QString vs, QString fs);
	void InitVAO();
	void InitVBO();
	void InitShader(QString vertexShaderPath, QString fragmentShaderPath);
	void DimensionTransformation(GLfloat source[], GLfloat target[][4]);
	void Begin();
	void Paint(GLfloat* ProjectionMatrix, GLfloat* ModelViewMatrix);
	float time = 0;
};