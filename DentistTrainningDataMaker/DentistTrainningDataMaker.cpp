#include "DentistTrainningDataMaker.h"

DentistTrainningDataMaker::DentistTrainningDataMaker(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// 事件連結
	connect(ui.actionLoadSTL,				SIGNAL(triggered()),			this,	SLOT(LoadSTL()));


	connect(ui.RenderTriangle_CheckBox,		SIGNAL(clicked()),			this,	SLOT(SetRenderTriangleBool()));
	connect(ui.RenderBorder_CheckBox,		SIGNAL(clicked()),			this,	SLOT(SetRenderBorderBool()));
	connect(ui.RenderPointDot_CheckBox,		SIGNAL(clicked()),			this,	SLOT(SetRenderPointCloudBool()));
}

// Render Options
void DentistTrainningDataMaker::LoadSTL()
{
	QString STLFileName = QFileDialog::getOpenFileName(this, tr("Load STL"), "./STLs", tr("STL (*.stl)"));
	system("cls");
	cout << "讀取位置：" << STLFileName.toStdString() << endl;

	ui.DisplayPanel->LoadSTLFile(STLFileName);
}

void DentistTrainningDataMaker::SetRenderTriangleBool()
{
	ui.DisplayPanel->SetRenderTriangleBool(ui.RenderTriangle_CheckBox->isChecked());
	ui.DisplayPanel->update();
}
void DentistTrainningDataMaker::SetRenderBorderBool()
{
	ui.DisplayPanel->SetRenderBorderBool(ui.RenderBorder_CheckBox->isChecked());
	ui.DisplayPanel->update();
}
void DentistTrainningDataMaker::SetRenderPointCloudBool()
{
	ui.DisplayPanel->SetRenderPointCloudBool(ui.RenderPointDot_CheckBox->isChecked());
	ui.DisplayPanel->update();
}
