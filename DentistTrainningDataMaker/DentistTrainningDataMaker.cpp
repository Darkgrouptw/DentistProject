#include "DentistTrainningDataMaker.h"

DentistTrainningDataMaker::DentistTrainningDataMaker(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// 事件連結
	connect(ui.actionLoadSTL,		SIGNAL(triggered()),			this,	SLOT(LoadSTL()));
}

void DentistTrainningDataMaker::LoadSTL()
{
	QString STLFileName = QFileDialog::getOpenFileName(this, tr("Load STL"), "./STLs", tr("STL (*.stl)"));
	system("cls");
	cout << "讀取位置：" << STLFileName.toStdString() << endl;

	ui.DisplayPanel->LoadSTLFile(STLFileName);
}
