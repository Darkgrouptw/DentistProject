#include "DentistProjectV2.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	// Windows 10 1809 的 bug
	system("chcp 65001");
	system("chcp 950");


	QApplication a(argc, argv);
	DentistProjectV2 w;
	w.show();
	return a.exec();
}
