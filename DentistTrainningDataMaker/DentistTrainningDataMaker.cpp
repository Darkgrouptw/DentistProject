#include "DentistTrainningDataMaker.h"

DentistTrainningDataMaker::DentistTrainningDataMaker(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// 事件連結
	connect(ui.actionLoadSTL,				SIGNAL(triggered()),			this,	SLOT(LoadSTL()));
	//connect(ui.TestFullScan,				SIGNAL(triggered()),			this,	SLOT(ScanData()));

	// 顯示事件
	connect(ui.RenderTriangle_CheckBox,		SIGNAL(clicked()),				this,	SLOT(SetRenderTriangleBool()));
	connect(ui.RenderBorder_CheckBox,		SIGNAL(clicked()),				this,	SLOT(SetRenderBorderBool()));
	connect(ui.RenderPointDot_CheckBox,		SIGNAL(clicked()),				this,	SLOT(SetRenderPointCloudBool()));

	// 藍芽部分
	connect(ui.BtnSearchCom,				SIGNAL(clicked()),				this,	SLOT(SearchCOM()));
	connect(ui.BtnConnectCOM,				SIGNAL(clicked()),				this,	SLOT(ConnectCOM()));
	connect(ui.BtnScanBLEDevice,			SIGNAL(clicked()),				this,	SLOT(ScanBLEDevice()));
	connect(ui.BtnConnectBLEDevice,			SIGNAL(clicked()),				this,	SLOT(ConnectBLEDeivce()));

	#pragma region 傳 UI 指標進去
	QVector<QObject*>		objList;

	objList.push_back(ui.BLEDeviceList);
	rawManager.bleManager.SendUIPointer(objList);
	#pragma endregion
}

// 其他事件
void DentistTrainningDataMaker::LoadSTL()
{
	QString STLFileName = QFileDialog::getOpenFileName(this, tr("Load STL"), "./STLs", tr("STL (*.stl)"), nullptr, QFileDialog::DontUseNativeDialog);
	system("cls");
	cout << "讀取位置：" << STLFileName.toStdString() << endl;

	ui.DisplayPanel->LoadSTLFile(STLFileName);
	ui.DisplayPanel->update();
}
void DentistTrainningDataMaker::ScanData()
{
	cout << "Test Scan" << endl;
}

// Render Options
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

// 藍芽事件
void DentistTrainningDataMaker::SearchCOM()
{
	QStringList COMListArray = rawManager.bleManager.GetCOMPortsArray();
	ui.COMList->clear();
	ui.COMList->addItems(COMListArray);
}
void DentistTrainningDataMaker::ConnectCOM()
{
	rawManager.bleManager.Initalize(ui.COMList->currentText());
}
void DentistTrainningDataMaker::ScanBLEDevice()
{
	if (rawManager.bleManager.IsInitialize())
		rawManager.bleManager.Scan();
	else
	{
		// 這邊要跳出說沒有建立連線，所以不能搜尋
	}
}
void DentistTrainningDataMaker::ConnectBLEDeivce()
{
	// cout << rawManager.bleManager.IsInitialize() << endl;
	rawManager.bleManager.Connect(ui.BLEDeviceList->currentIndex());
}
