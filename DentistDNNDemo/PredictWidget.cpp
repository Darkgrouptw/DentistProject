#include "PredictWidget.h"

PredictWidget::PredictWidget(QWidget* parent = 0) : QOpenGLWidget(parent)
{
}
PredictWidget::~PredictWidget()
{
}

// 繪畫相關函式
void PredictWidget::initializeGL()
{
#pragma region 初始化參數
	initializeOpenGLFunctions();
	glClearColor(0.5f, 0.5, 0.5f, 1);

	// 連結 Shader
	Program = new QOpenGLShaderProgram();
	Program->addCacheableShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/Plane.vsh");
	Program->addCacheableShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/Plane.fsh");
	Program->link();
#pragma endregion
#pragma region Vertex & UV
	QVector<QVector3D> plane_vertices;
	plane_vertices.push_back(QVector3D(-1, -1, 0));
	plane_vertices.push_back(QVector3D(1, -1, 0));
	plane_vertices.push_back(QVector3D(1, 1, 0));
	plane_vertices.push_back(QVector3D(-1, 1, 0));

	QVector<QVector2D> plane_uvs;
	plane_uvs.push_back(QVector2D(0, 1));
	plane_uvs.push_back(QVector2D(1, 1));
	plane_uvs.push_back(QVector2D(1, 0));
	plane_uvs.push_back(QVector2D(0, 0));
#pragma endregion
#pragma region Bind Buffer
	// VertexBuffer
	glGenBuffers(1, &VertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, plane_vertices.size() * sizeof(QVector3D), plane_vertices.constData(), GL_STATIC_DRAW);

	// UV
	glGenBuffers(1, &UVBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, UVBuffer);
	glBufferData(GL_ARRAY_BUFFER, plane_uvs.size() * sizeof(QVector2D), plane_uvs.constData(), GL_STATIC_DRAW);
#pragma endregion
}
void PredictWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (OtherSideTexture != NULL)
	{
		Program->bind();

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, UVBuffer);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		Program->setUniformValue("texture", 0);
		Program->setUniformValue("tempuv", 1.0f);
		OtherSideTexture->bind(0);

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		OtherSideTexture->release();
		Program->release();

		glLineWidth(2.0f);

		glPushMatrix();
		glBegin(GL_LINES);


		//glColor3f(163.0 / 255.0, 126.0 / 255.0, 204.0 / 255.0);
		glColor3f(1.0, 0.0, 0.0);

		for (float i = -1.0; i < 1.0; i += 0.1) {
			glVertex2f(i, -Yvalue);
			glVertex2f(i + 0.035, -Yvalue);
		}

		glColor3f(1.0, 1.0, 1.0);

		glEnd();
		glPopMatrix();
	}



}

// 外部呼叫函式
void PredictWidget::ProcessImg(Mat FullMat, int SliderValue, float Y)
{
#pragma region 轉成 QOpenGLTexture

	Xvalue = ((SliderValue / 250.0f) - 0.5f) * 2.0f;
	Yvalue = Y;

	if (OtherSideTexture != NULL)
	{
		delete OtherSideTexture;
	}
	OtherSideTexture = new QOpenGLTexture(Mat2QImage(FullMat, CV_8UC3));
#pragma endregion
}
// Helper Function
QImage PredictWidget::Mat2QImage(cv::Mat const& src, int Type)
{
	cv::Mat temp;												// make the same cv::Mat
	if (Type == CV_8UC3)
		cvtColor(src, temp, CV_BGR2RGB);						// cvtColor Makes a copt, that what i need
	else if (Type == CV_32F)
	{
		src.convertTo(temp, CV_8U, 255);
		cvtColor(temp, temp, CV_GRAY2RGB);						// cvtColor Makes a copt, that what i need
	}
	QImage dest((const uchar *)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
	dest.bits();												// enforce deep copy, see documentation 
																// of QImage::QImage ( const uchar * data, int width, int height, Format format )
	return dest;
}
