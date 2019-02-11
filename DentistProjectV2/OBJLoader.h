#pragma once
#include <QFile>
#include <QTextStream>
#include <QIODevice>

#include <QVector>
#include <QVector2D>
#include <QVector3D>
class OBJLoader
{
public:
	static bool loadOBJ(
		QString,
		QVector<QVector3D> &,
		QVector<QVector2D> &,
		QVector<QVector3D> &,
		QVector<unsigned int> &,
		QVector<QString> &
		);

	static bool loadMTL(
		QString,
		QVector<QVector3D> &,
		QVector<QVector3D> &,
		QVector<QVector3D> &,
		QVector<QString> &,
		QString &);
};

