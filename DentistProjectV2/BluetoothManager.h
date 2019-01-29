#pragma once
/*
藍芽的 Manager
*/
#include "BluetoothLeDevice.h"

#using <System.dll> 
#include <iostream>
#include <string>
#include <cassert>
#include <functional>

#include <QtMath>
#include <QVector3D>
#include <QQuaternion>
#include <QTextCodec>
#include <QString>
#include <QVector>
#include <QStringList>
#include <QLabel>
#include <QComboBox>
#include <QMessageBox>
#include <QMainWindow>

// 重新命名 Namespace (因為明明就是藍芽模組，硬要寫成手套= =)
namespace LibBLE = LibGlove;
using namespace std;
using namespace System::IO::Ports;
using namespace System::Collections::Generic;
using namespace System::Threading;

class BluetoothManager
{
public:
	BluetoothManager();
	~BluetoothManager();

	//////////////////////////////////////////////////////////////////////////
	// 外部呼叫函數
	//////////////////////////////////////////////////////////////////////////
	void				SendUIPointer(QVector<QObject*>);							// 這個 Function 是用來，當藍芽有新狀態時，要去更動外面的 UI 時，必須要先拿到外面的 UI 指標，所以要先送來一份
	QStringList			GetCOMPortsArray();
	void				Initalize(QString);
	void				Scan();
	void				Connect(int);
	void				SetConnectDirectly(string, string);							// 一鍵連結會用到，直接給 Name & Mac Address 連結
	bool				IsInitialize();
	void				SetOffsetQuat();
	QVector3D			GetAngle();

private:
	//////////////////////////////////////////////////////////////////////////
	// 藍芽 & Callback
	//////////////////////////////////////////////////////////////////////////
	LibBLE::BluetoothLeDevice *device;
	QVector<LibBLE::DeviceInfo*>	deviceInfoList;
	void				Callback_DeviceInitDone(string);
	void				Callback_DeviceCloseDone(string);
	void				Callback_DeviceDiscovered(LibBLE::DeviceInfo*);
	void				Callback_EstablishLinkDone(LibBLE::DeviceInfo*, unsigned char);
	void				Callback_TerminateLinkDone(LibGlove::DeviceInfo*, unsigned char, bool);
	void				Callback_QuaternionRotationChanged(LibGlove::DeviceInfo*, float[]);

	//////////////////////////////////////////////////////////////////////////
	// UI pointer
	//////////////////////////////////////////////////////////////////////////
	QLabel*				BLEStatus;
	QLabel*				EularText;
	QMainWindow*		MainWindow;
	QComboBox*			bleTextList;

	//////////////////////////////////////////////////////////////////////////
	// 九軸相關
	//////////////////////////////////////////////////////////////////////////
	QQuaternion			CurrentQuat;
	QQuaternion			OffsetQuat;
	float				AngleX = 0;
	float				AngleY = 0;
	float				AngleZ = 0;

	//////////////////////////////////////////////////////////////////////////
	// Other
	//////////////////////////////////////////////////////////////////////////
	bool				IsDirectConnected = false;								// 這邊要做一件事是直接 Connect 當裝置 Init 完之後，但一般不會進這裡
	LibBLE::DeviceInfo* BLEDirectInfo = new LibBLE::DeviceInfo;
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	void				MarshalString(System::String^, string&);
};