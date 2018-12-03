#include "DentistProject.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	DentistProject w;
	w.show();
	return a.exec();
}
