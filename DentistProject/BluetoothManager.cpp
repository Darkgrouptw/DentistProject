﻿#include "BluetoothManager.h"

BluetoothManager::BluetoothManager()
{
	#pragma region 初始化 callback
	device = new LibBLE::BluetoothLeDevice();
	device->RegisterCallback_DeviceInitDone(		bind(&BluetoothManager::Callback_DeviceInitDone,	this, placeholders::_1));								// 初始化 (打開 COM 的時候)
	device->RegisterCallback_DeviceCloseDone(		bind(&BluetoothManager::Callback_DeviceCloseDone,	this, placeholders::_1));								// 關閉 COM
	device->RegisterCallback_DeviceDiscovered(		bind(&BluetoothManager::Callback_DeviceDiscovered,	this, placeholders::_1));								// Scan 到其他裝置的時候
	device->RegisterCallback_EstablishLinkDone(		bind(&BluetoothManager::Callback_EstablishLinkDone,	this, std::placeholders::_1, std::placeholders::_2));	// 建立連線的判斷
	device->RegisterCallback_TerminateLinkDone(		bind(&BluetoothManager::Callback_TerminateLinkDone, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));	//這個是中斷連線
	device->RegisterCallback_QuaternionRotationChanged(bind(&BluetoothManager::Callback_QuaternionRotationChanged, this, std::placeholders::_1, std::placeholders::_2));				// 拿 Quaternion 的資料

	/*
	這邊是沒有用到的 Callback 可能後面會用到
	device->RegisterCallback_HandleCheckNotifyStatus(std::bind(&BluetoothManager::Callback_HandleCheckNotifyStatus, this, std::placeholders::_1, std::placeholders::_2));
	device->RegisterCallback_HandleChangeNotifyStatus(std::bind(&BluetoothManager::Callback_HandleChangeNotifyStatus, this, std::placeholders::_1, std::placeholders::_2));
	device->RegisterCallback_AccelerationChanged(std::bind(&BluetoothManager::Callback_AccelerationChanged, this, std::placeholders::_1, std::placeholders::_2));
	device->RegisterCallback_QuaternionRotationChanged(std::bind(&BluetoothManager::Callback_QuaternionRotationChanged, this, std::placeholders::_1, std::placeholders::_2));
	device->RegisterCallback_EulerRotationChanged(std::bind(&BluetoothManager::Callback_EulerRotationChanged, this, std::placeholders::_1, std::placeholders::_2));
	device->RegisterCallback_FlexionChanged(std::bind(&BluetoothManager::Callback_FlexionChanged, this, std::placeholders::_1, std::placeholders::_2));
	*/
	#pragma endregion
}
BluetoothManager::~BluetoothManager()
{
	if (device->IsEstablished())
	{
		device->Terminate();
		while (device->IsEstablished())
		{
			Thread::Sleep(100);
		}
	}
	if (device->IsInitialized())
		device->Close();
	delete device;
}

void BluetoothManager::SendUIPointer(QVector<QObject*> objList)
{
	// 確認是不是有多傳，忘了改的
	assert(objList.size() == 4);
	BLEStatus		= (QLabel*)		objList[0];
	QuaternionText	= (QLabel*)		objList[1];
	MainWindow		= (QMainWindow*) objList[2];
	bleTextList		= (QComboBox*)	objList[3];
}
QStringList BluetoothManager::GetCOMPortsArray()
{
	cli::array<System::String^>^ sArray = SerialPort::GetPortNames();
	string TempStr;
	QStringList outputStr;
	for (int i = 0; i < sArray->Length; i++)
	{
		MarshalString(sArray[i], TempStr);
		outputStr.push_back(QString::fromStdString(TempStr));
	}
	return outputStr;
}
void BluetoothManager::Initalize(QString str)
{
	// 判斷前面有沒有連過
	if (IsInitialize())
		device->Close();

	bool IsOpen = device->Initialize(str.toStdString());
	if (IsOpen)
		cout << "成功打開 COM" << endl;
	else
		cout << "打不開 COM" << endl;
	
}
void BluetoothManager::Scan()
{
	// 先暫停前面一次的結果
	device->ScanCancel();

	// 清空可用的 List
	deviceInfoList.clear();

	// 掃描
	device->Scan();
}
void BluetoothManager::Connect(int index)
{
	if (device->IsInitialized() && !device->IsEstablished())
	{
		device->ScanCancel();

		Thread::Sleep(100);

		device->Establish(deviceInfoList[index]);		
	}
	else if (device->IsInitialized())
		device->Terminate();
}

bool BluetoothManager::IsInitialize()
{
	return device->IsInitialized();
}

//////////////////////////////////////////////////////////////////////////
// 藍芽 & Callback
//////////////////////////////////////////////////////////////////////////
void BluetoothManager::Callback_DeviceInitDone(string com)
{
	cout << "藍芽初始化: " <<  com << " " << (device->IsInitialized()? "成功!": "失敗!") << endl;
	BLEStatus->setText(codec->toUnicode("藍芽狀態：初始化成功"));
}
void BluetoothManager::Callback_DeviceCloseDone(string com)
{
	cout << "藍芽關閉完成: " << com << " " << (!device->IsInitialized() ? "成功!" : "失敗!") << endl;
	BLEStatus->setText(codec->toUnicode("藍芽狀態：未連結"));
}

void BluetoothManager::Callback_DeviceDiscovered(LibBLE::DeviceInfo* deviceInfo)
{
	cout << "找到裝置: " << deviceInfo->DeviceName << " " << deviceInfo->DeviceAddress << endl;
	BLEStatus->setText(codec->toUnicode("藍芽狀態：搜尋裝置"));

	deviceInfoList.push_back(deviceInfo);
	bleTextList->addItem(QString::fromStdString(deviceInfo->DeviceName + "(" + deviceInfo->DeviceAddress + ")"));
}
void BluetoothManager::Callback_EstablishLinkDone(LibBLE::DeviceInfo* deviceInfo, unsigned char status)
{
	if (status == LibBLE::BluetoothLeDevice::StatusCodes::Success)
	{
		cout << "連線成功: " << deviceInfo->DeviceName << " " << deviceInfo->DeviceAddress << endl;
		device->StartNotification();
	}
	else
		cout << "連線失敗: " << deviceInfo->DeviceName << " " << deviceInfo->DeviceAddress << endl;

}
void BluetoothManager::Callback_TerminateLinkDone(LibGlove::DeviceInfo* deviceInfo, unsigned char status, bool byHost)
{
	if (status == LibGlove::BluetoothLeDevice::StatusCodes::Success)
		cout << "斷開連線成功: " << deviceInfo->DeviceName << " " << deviceInfo->DeviceAddress << endl;
	else
		cout << "斷開連線失敗: " << deviceInfo->DeviceName << " " << deviceInfo->DeviceAddress << endl;
}
void BluetoothManager::Callback_QuaternionRotationChanged(LibGlove::DeviceInfo*, float quat[])
{
	// 改狀態
	BLEStatus->setText(codec->toUnicode("藍芽狀態：傳輸資料中"));

	// 旋轉結果
	string Quaternion_Output = "Ｗ： " + to_string(quat[0]) + "\nＸ： " + to_string(quat[1]) + "\nＹ： " + to_string(quat[2]) + "\nＺ： " + to_string(quat[3]);
	QuaternionText->setText(codec->toUnicode(Quaternion_Output.c_str()));
}

//////////////////////////////////////////////////////////////////////////
// Helper Function
//////////////////////////////////////////////////////////////////////////
void BluetoothManager::MarshalString(System::String ^ s, string& os) 
{
	using namespace System::Runtime::InteropServices;
	const char* chars =
		(const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	os = chars;
	Marshal::FreeHGlobal(System::IntPtr((void*)chars));
}