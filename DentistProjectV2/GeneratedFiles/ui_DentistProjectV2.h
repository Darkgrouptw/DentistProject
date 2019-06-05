/********************************************************************************
** Form generated from reading UI file 'DentistProjectV2.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DENTISTPROJECTV2_H
#define UI_DENTISTPROJECTV2_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>
#include "openglwidget.h"

QT_BEGIN_NAMESPACE

class Ui_DentistProjectV2Class
{
public:
    QWidget *centralWidget;
    OpenGLWidget *DisplayPanel;
    QGroupBox *ScanResult;
    QLabel *ImageResult;
    QLabel *BorderDetectionResult;
    QLabel *ImageResultText;
    QLabel *BorderDetectionResultText;
    QSlider *ScanNumSlider;
    QLabel *ScanNum_Min;
    QLabel *ScanNum_Max;
    QLabel *ScanNum_Value;
    QLabel *NetworkResultText;
    QLabel *NetworkResult;
    QTabWidget *tabWidget;
    QWidget *Tab_Deivce;
    QGroupBox *BLEDeviceBox;
    QPushButton *BtnConnectCOM;
    QPushButton *BtnScanBLEDevice;
    QPushButton *BtnConnectBLEDevice;
    QPushButton *BtnSearchCom;
    QComboBox *BLEDeviceList;
    QComboBox *COMList;
    QGroupBox *BLEDeviceInfoBox;
    QLabel *EularText;
    QLabel *BLEStatus;
    QGroupBox *ResetRotationBox;
    QPushButton *ResetRotationMode;
    QPushButton *GyroscopeResetToZero;
    QGroupBox *BLETestingBox;
    QPushButton *PointCloudAlignmentTestBtn;
    QPushButton *BLEConnect_OneBtn;
    QWidget *Tab_OCT;
    QGroupBox *OCTNormalSettingBox;
    QLineEdit *SaveLocationText;
    QLabel *SaveLocationLabel;
    QPushButton *SaveLocationBtn;
    QCheckBox *AutoSaveSingleRawDataWhileScan_CheckBox;
    QPushButton *ScanButton;
    QCheckBox *AutoSaveImageWhileScan_CheckBox;
    QCheckBox *AutoSaveMultiRawDataWhileScan_CheckBox;
    QCheckBox *AutoDeleteShakeData_CheckBox;
    QPushButton *ScanOnceButton;
    QGroupBox *OCTTestingBox;
    QPushButton *RawDataToImage;
    QPushButton *EasyBorderDetect;
    QPushButton *SingleImageShakeTestButton;
    QPushButton *SegNetTestButton;
    QPushButton *BeepSoundTestButton;
    QPushButton *MultiImageShakeTestButton;
    QPushButton *SlimLabViewRawData;
    QWidget *Tab_PC;
    QGroupBox *AlignBox;
    QPushButton *AlignLastTwoPCButton;
    QPushButton *CombineLastTwoPCButton;
    QPushButton *CombineAllPCButton;
    QPushButton *AlignmentAllPCTest;
    QPushButton *PassScanDataToPC;
    QPushButton *AveragePCErrorTest;
    QGroupBox *PCInfoBox;
    QLabel *ChoosePCIndexText;
    QComboBox *PCIndex;
    QGroupBox *PCOperationBox;
    QPushButton *LoadPCButton;
    QPushButton *SavePCButton;
    QPushButton *DeletePCButton;
    QWidget *Tab_Network;
    QGroupBox *NetworkDataOperationBox;
    QPushButton *DataGenerationBtn;
    QGroupBox *NetworkDataTestBox;
    QPushButton *VolumeRenderingTestBtn;
    QPushButton *PredictResultTestingBtn;
    QWidget *StateWidget;
    QLabel *OtherSideResult;
    QLabel *NetworkResult_OtherSide;
    QGroupBox *RenderGroupBox;
    QComboBox *OCTViewDir;
    QLabel *OCTViewDirText;

    void setupUi(QMainWindow *DentistProjectV2Class)
    {
        if (DentistProjectV2Class->objectName().isEmpty())
            DentistProjectV2Class->setObjectName(QStringLiteral("DentistProjectV2Class"));
        DentistProjectV2Class->resize(1600, 900);
        centralWidget = new QWidget(DentistProjectV2Class);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        DisplayPanel = new OpenGLWidget(centralWidget);
        DisplayPanel->setObjectName(QStringLiteral("DisplayPanel"));
        DisplayPanel->setGeometry(QRect(0, 0, 900, 900));
        ScanResult = new QGroupBox(centralWidget);
        ScanResult->setObjectName(QStringLiteral("ScanResult"));
        ScanResult->setGeometry(QRect(900, 0, 581, 561));
        ImageResult = new QLabel(ScanResult);
        ImageResult->setObjectName(QStringLiteral("ImageResult"));
        ImageResult->setGeometry(QRect(10, 40, 550, 135));
        ImageResult->setStyleSheet(QStringLiteral(""));
        ImageResult->setScaledContents(true);
        BorderDetectionResult = new QLabel(ScanResult);
        BorderDetectionResult->setObjectName(QStringLiteral("BorderDetectionResult"));
        BorderDetectionResult->setGeometry(QRect(10, 200, 550, 135));
        BorderDetectionResult->setStyleSheet(QStringLiteral(""));
        BorderDetectionResult->setScaledContents(true);
        ImageResultText = new QLabel(ScanResult);
        ImageResultText->setObjectName(QStringLiteral("ImageResultText"));
        ImageResultText->setGeometry(QRect(10, 20, 101, 16));
        BorderDetectionResultText = new QLabel(ScanResult);
        BorderDetectionResultText->setObjectName(QStringLiteral("BorderDetectionResultText"));
        BorderDetectionResultText->setGeometry(QRect(10, 180, 151, 16));
        ScanNumSlider = new QSlider(ScanResult);
        ScanNumSlider->setObjectName(QStringLiteral("ScanNumSlider"));
        ScanNumSlider->setEnabled(false);
        ScanNumSlider->setGeometry(QRect(20, 510, 501, 22));
        ScanNumSlider->setMinimum(60);
        ScanNumSlider->setMaximum(200);
        ScanNumSlider->setValue(60);
        ScanNumSlider->setOrientation(Qt::Horizontal);
        ScanNum_Min = new QLabel(ScanResult);
        ScanNum_Min->setObjectName(QStringLiteral("ScanNum_Min"));
        ScanNum_Min->setGeometry(QRect(20, 540, 21, 16));
        ScanNum_Max = new QLabel(ScanResult);
        ScanNum_Max->setObjectName(QStringLiteral("ScanNum_Max"));
        ScanNum_Max->setGeometry(QRect(510, 540, 21, 16));
        ScanNum_Value = new QLabel(ScanResult);
        ScanNum_Value->setObjectName(QStringLiteral("ScanNum_Value"));
        ScanNum_Value->setGeometry(QRect(540, 510, 21, 16));
        NetworkResultText = new QLabel(ScanResult);
        NetworkResultText->setObjectName(QStringLiteral("NetworkResultText"));
        NetworkResultText->setEnabled(true);
        NetworkResultText->setGeometry(QRect(10, 340, 111, 16));
        NetworkResult = new QLabel(ScanResult);
        NetworkResult->setObjectName(QStringLiteral("NetworkResult"));
        NetworkResult->setEnabled(true);
        NetworkResult->setGeometry(QRect(10, 360, 550, 135));
        NetworkResult->setStyleSheet(QStringLiteral(""));
        NetworkResult->setScaledContents(true);
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(900, 570, 710, 331));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy);
        Tab_Deivce = new QWidget();
        Tab_Deivce->setObjectName(QStringLiteral("Tab_Deivce"));
        BLEDeviceBox = new QGroupBox(Tab_Deivce);
        BLEDeviceBox->setObjectName(QStringLiteral("BLEDeviceBox"));
        BLEDeviceBox->setEnabled(true);
        BLEDeviceBox->setGeometry(QRect(10, 10, 521, 101));
        BtnConnectCOM = new QPushButton(BLEDeviceBox);
        BtnConnectCOM->setObjectName(QStringLiteral("BtnConnectCOM"));
        BtnConnectCOM->setGeometry(QRect(410, 20, 91, 23));
        BtnScanBLEDevice = new QPushButton(BLEDeviceBox);
        BtnScanBLEDevice->setObjectName(QStringLiteral("BtnScanBLEDevice"));
        BtnScanBLEDevice->setGeometry(QRect(310, 60, 91, 23));
        BtnConnectBLEDevice = new QPushButton(BLEDeviceBox);
        BtnConnectBLEDevice->setObjectName(QStringLiteral("BtnConnectBLEDevice"));
        BtnConnectBLEDevice->setGeometry(QRect(410, 60, 91, 23));
        BtnSearchCom = new QPushButton(BLEDeviceBox);
        BtnSearchCom->setObjectName(QStringLiteral("BtnSearchCom"));
        BtnSearchCom->setGeometry(QRect(310, 20, 91, 23));
        BLEDeviceList = new QComboBox(BLEDeviceBox);
        BLEDeviceList->setObjectName(QStringLiteral("BLEDeviceList"));
        BLEDeviceList->setGeometry(QRect(10, 60, 291, 22));
        COMList = new QComboBox(BLEDeviceBox);
        COMList->setObjectName(QStringLiteral("COMList"));
        COMList->setGeometry(QRect(10, 20, 291, 22));
        BLEDeviceInfoBox = new QGroupBox(Tab_Deivce);
        BLEDeviceInfoBox->setObjectName(QStringLiteral("BLEDeviceInfoBox"));
        BLEDeviceInfoBox->setGeometry(QRect(20, 120, 181, 151));
        EularText = new QLabel(BLEDeviceInfoBox);
        EularText->setObjectName(QStringLiteral("EularText"));
        EularText->setGeometry(QRect(10, 60, 121, 61));
        QFont font;
        font.setPointSize(12);
        EularText->setFont(font);
        EularText->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);
        BLEStatus = new QLabel(BLEDeviceInfoBox);
        BLEStatus->setObjectName(QStringLiteral("BLEStatus"));
        BLEStatus->setGeometry(QRect(10, 20, 161, 31));
        BLEStatus->setFont(font);
        ResetRotationBox = new QGroupBox(Tab_Deivce);
        ResetRotationBox->setObjectName(QStringLiteral("ResetRotationBox"));
        ResetRotationBox->setGeometry(QRect(210, 120, 151, 141));
        ResetRotationMode = new QPushButton(ResetRotationBox);
        ResetRotationMode->setObjectName(QStringLiteral("ResetRotationMode"));
        ResetRotationMode->setGeometry(QRect(10, 20, 131, 31));
        ResetRotationMode->setAutoDefault(false);
        ResetRotationMode->setFlat(false);
        GyroscopeResetToZero = new QPushButton(ResetRotationBox);
        GyroscopeResetToZero->setObjectName(QStringLiteral("GyroscopeResetToZero"));
        GyroscopeResetToZero->setGeometry(QRect(10, 60, 131, 31));
        GyroscopeResetToZero->setAutoDefault(false);
        GyroscopeResetToZero->setFlat(false);
        BLETestingBox = new QGroupBox(Tab_Deivce);
        BLETestingBox->setObjectName(QStringLiteral("BLETestingBox"));
        BLETestingBox->setEnabled(true);
        BLETestingBox->setGeometry(QRect(540, 10, 151, 271));
        PointCloudAlignmentTestBtn = new QPushButton(BLETestingBox);
        PointCloudAlignmentTestBtn->setObjectName(QStringLiteral("PointCloudAlignmentTestBtn"));
        PointCloudAlignmentTestBtn->setEnabled(true);
        PointCloudAlignmentTestBtn->setGeometry(QRect(10, 20, 131, 23));
        PointCloudAlignmentTestBtn->setCheckable(false);
        BLEConnect_OneBtn = new QPushButton(Tab_Deivce);
        BLEConnect_OneBtn->setObjectName(QStringLiteral("BLEConnect_OneBtn"));
        BLEConnect_OneBtn->setEnabled(true);
        BLEConnect_OneBtn->setGeometry(QRect(370, 130, 151, 111));
        tabWidget->addTab(Tab_Deivce, QString());
        Tab_OCT = new QWidget();
        Tab_OCT->setObjectName(QStringLiteral("Tab_OCT"));
        OCTNormalSettingBox = new QGroupBox(Tab_OCT);
        OCTNormalSettingBox->setObjectName(QStringLiteral("OCTNormalSettingBox"));
        OCTNormalSettingBox->setGeometry(QRect(10, 10, 521, 271));
        SaveLocationText = new QLineEdit(OCTNormalSettingBox);
        SaveLocationText->setObjectName(QStringLiteral("SaveLocationText"));
        SaveLocationText->setEnabled(false);
        SaveLocationText->setGeometry(QRect(10, 40, 421, 20));
        SaveLocationLabel = new QLabel(OCTNormalSettingBox);
        SaveLocationLabel->setObjectName(QStringLiteral("SaveLocationLabel"));
        SaveLocationLabel->setGeometry(QRect(10, 20, 111, 16));
        SaveLocationBtn = new QPushButton(OCTNormalSettingBox);
        SaveLocationBtn->setObjectName(QStringLiteral("SaveLocationBtn"));
        SaveLocationBtn->setGeometry(QRect(440, 40, 75, 23));
        AutoSaveSingleRawDataWhileScan_CheckBox = new QCheckBox(OCTNormalSettingBox);
        AutoSaveSingleRawDataWhileScan_CheckBox->setObjectName(QStringLiteral("AutoSaveSingleRawDataWhileScan_CheckBox"));
        AutoSaveSingleRawDataWhileScan_CheckBox->setGeometry(QRect(10, 70, 191, 16));
        AutoSaveSingleRawDataWhileScan_CheckBox->setChecked(false);
        ScanButton = new QPushButton(OCTNormalSettingBox);
        ScanButton->setObjectName(QStringLiteral("ScanButton"));
        ScanButton->setGeometry(QRect(270, 110, 231, 151));
        AutoSaveImageWhileScan_CheckBox = new QCheckBox(OCTNormalSettingBox);
        AutoSaveImageWhileScan_CheckBox->setObjectName(QStringLiteral("AutoSaveImageWhileScan_CheckBox"));
        AutoSaveImageWhileScan_CheckBox->setGeometry(QRect(10, 130, 191, 16));
        AutoSaveImageWhileScan_CheckBox->setChecked(false);
        AutoSaveMultiRawDataWhileScan_CheckBox = new QCheckBox(OCTNormalSettingBox);
        AutoSaveMultiRawDataWhileScan_CheckBox->setObjectName(QStringLiteral("AutoSaveMultiRawDataWhileScan_CheckBox"));
        AutoSaveMultiRawDataWhileScan_CheckBox->setGeometry(QRect(10, 100, 191, 16));
        AutoSaveMultiRawDataWhileScan_CheckBox->setChecked(true);
        AutoDeleteShakeData_CheckBox = new QCheckBox(OCTNormalSettingBox);
        AutoDeleteShakeData_CheckBox->setObjectName(QStringLiteral("AutoDeleteShakeData_CheckBox"));
        AutoDeleteShakeData_CheckBox->setGeometry(QRect(10, 160, 191, 16));
        AutoDeleteShakeData_CheckBox->setChecked(true);
        ScanOnceButton = new QPushButton(OCTNormalSettingBox);
        ScanOnceButton->setObjectName(QStringLiteral("ScanOnceButton"));
        ScanOnceButton->setGeometry(QRect(10, 200, 151, 61));
        OCTTestingBox = new QGroupBox(Tab_OCT);
        OCTTestingBox->setObjectName(QStringLiteral("OCTTestingBox"));
        OCTTestingBox->setEnabled(true);
        OCTTestingBox->setGeometry(QRect(540, 10, 151, 271));
        RawDataToImage = new QPushButton(OCTTestingBox);
        RawDataToImage->setObjectName(QStringLiteral("RawDataToImage"));
        RawDataToImage->setGeometry(QRect(10, 20, 131, 23));
        EasyBorderDetect = new QPushButton(OCTTestingBox);
        EasyBorderDetect->setObjectName(QStringLiteral("EasyBorderDetect"));
        EasyBorderDetect->setGeometry(QRect(10, 50, 131, 23));
        SingleImageShakeTestButton = new QPushButton(OCTTestingBox);
        SingleImageShakeTestButton->setObjectName(QStringLiteral("SingleImageShakeTestButton"));
        SingleImageShakeTestButton->setGeometry(QRect(10, 80, 131, 23));
        SegNetTestButton = new QPushButton(OCTTestingBox);
        SegNetTestButton->setObjectName(QStringLiteral("SegNetTestButton"));
        SegNetTestButton->setGeometry(QRect(10, 240, 131, 23));
        BeepSoundTestButton = new QPushButton(OCTTestingBox);
        BeepSoundTestButton->setObjectName(QStringLiteral("BeepSoundTestButton"));
        BeepSoundTestButton->setGeometry(QRect(10, 210, 131, 23));
        MultiImageShakeTestButton = new QPushButton(OCTTestingBox);
        MultiImageShakeTestButton->setObjectName(QStringLiteral("MultiImageShakeTestButton"));
        MultiImageShakeTestButton->setEnabled(false);
        MultiImageShakeTestButton->setGeometry(QRect(10, 110, 131, 23));
        MultiImageShakeTestButton->setCheckable(false);
        SlimLabViewRawData = new QPushButton(OCTTestingBox);
        SlimLabViewRawData->setObjectName(QStringLiteral("SlimLabViewRawData"));
        SlimLabViewRawData->setEnabled(true);
        SlimLabViewRawData->setGeometry(QRect(10, 180, 131, 23));
        SlimLabViewRawData->setCheckable(false);
        tabWidget->addTab(Tab_OCT, QString());
        Tab_PC = new QWidget();
        Tab_PC->setObjectName(QStringLiteral("Tab_PC"));
        AlignBox = new QGroupBox(Tab_PC);
        AlignBox->setObjectName(QStringLiteral("AlignBox"));
        AlignBox->setGeometry(QRect(540, 10, 151, 281));
        AlignLastTwoPCButton = new QPushButton(AlignBox);
        AlignLastTwoPCButton->setObjectName(QStringLiteral("AlignLastTwoPCButton"));
        AlignLastTwoPCButton->setGeometry(QRect(10, 20, 131, 23));
        CombineLastTwoPCButton = new QPushButton(AlignBox);
        CombineLastTwoPCButton->setObjectName(QStringLiteral("CombineLastTwoPCButton"));
        CombineLastTwoPCButton->setEnabled(true);
        CombineLastTwoPCButton->setGeometry(QRect(10, 50, 131, 23));
        CombineAllPCButton = new QPushButton(AlignBox);
        CombineAllPCButton->setObjectName(QStringLiteral("CombineAllPCButton"));
        CombineAllPCButton->setEnabled(true);
        CombineAllPCButton->setGeometry(QRect(10, 80, 131, 23));
        AlignmentAllPCTest = new QPushButton(AlignBox);
        AlignmentAllPCTest->setObjectName(QStringLiteral("AlignmentAllPCTest"));
        AlignmentAllPCTest->setEnabled(true);
        AlignmentAllPCTest->setGeometry(QRect(10, 110, 131, 41));
        PassScanDataToPC = new QPushButton(AlignBox);
        PassScanDataToPC->setObjectName(QStringLiteral("PassScanDataToPC"));
        PassScanDataToPC->setEnabled(true);
        PassScanDataToPC->setGeometry(QRect(10, 160, 131, 41));
        AveragePCErrorTest = new QPushButton(AlignBox);
        AveragePCErrorTest->setObjectName(QStringLiteral("AveragePCErrorTest"));
        AveragePCErrorTest->setEnabled(true);
        AveragePCErrorTest->setGeometry(QRect(10, 232, 131, 41));
        PCInfoBox = new QGroupBox(Tab_PC);
        PCInfoBox->setObjectName(QStringLiteral("PCInfoBox"));
        PCInfoBox->setGeometry(QRect(10, 10, 521, 281));
        ChoosePCIndexText = new QLabel(PCInfoBox);
        ChoosePCIndexText->setObjectName(QStringLiteral("ChoosePCIndexText"));
        ChoosePCIndexText->setGeometry(QRect(20, 20, 61, 16));
        PCIndex = new QComboBox(PCInfoBox);
        PCIndex->setObjectName(QStringLiteral("PCIndex"));
        PCIndex->setGeometry(QRect(80, 20, 131, 22));
        PCOperationBox = new QGroupBox(PCInfoBox);
        PCOperationBox->setObjectName(QStringLiteral("PCOperationBox"));
        PCOperationBox->setGeometry(QRect(10, 210, 301, 61));
        LoadPCButton = new QPushButton(PCOperationBox);
        LoadPCButton->setObjectName(QStringLiteral("LoadPCButton"));
        LoadPCButton->setGeometry(QRect(10, 20, 81, 31));
        SavePCButton = new QPushButton(PCOperationBox);
        SavePCButton->setObjectName(QStringLiteral("SavePCButton"));
        SavePCButton->setGeometry(QRect(100, 20, 91, 31));
        DeletePCButton = new QPushButton(PCOperationBox);
        DeletePCButton->setObjectName(QStringLiteral("DeletePCButton"));
        DeletePCButton->setGeometry(QRect(200, 20, 91, 31));
        tabWidget->addTab(Tab_PC, QString());
        Tab_Network = new QWidget();
        Tab_Network->setObjectName(QStringLiteral("Tab_Network"));
        NetworkDataOperationBox = new QGroupBox(Tab_Network);
        NetworkDataOperationBox->setObjectName(QStringLiteral("NetworkDataOperationBox"));
        NetworkDataOperationBox->setGeometry(QRect(10, 10, 151, 91));
        DataGenerationBtn = new QPushButton(NetworkDataOperationBox);
        DataGenerationBtn->setObjectName(QStringLiteral("DataGenerationBtn"));
        DataGenerationBtn->setGeometry(QRect(10, 20, 131, 61));
        NetworkDataTestBox = new QGroupBox(Tab_Network);
        NetworkDataTestBox->setObjectName(QStringLiteral("NetworkDataTestBox"));
        NetworkDataTestBox->setGeometry(QRect(530, 10, 151, 281));
        VolumeRenderingTestBtn = new QPushButton(NetworkDataTestBox);
        VolumeRenderingTestBtn->setObjectName(QStringLiteral("VolumeRenderingTestBtn"));
        VolumeRenderingTestBtn->setEnabled(false);
        VolumeRenderingTestBtn->setGeometry(QRect(10, 250, 131, 23));
        PredictResultTestingBtn = new QPushButton(NetworkDataTestBox);
        PredictResultTestingBtn->setObjectName(QStringLiteral("PredictResultTestingBtn"));
        PredictResultTestingBtn->setGeometry(QRect(10, 20, 131, 61));
        tabWidget->addTab(Tab_Network, QString());
        StateWidget = new QWidget(centralWidget);
        StateWidget->setObjectName(QStringLiteral("StateWidget"));
        StateWidget->setGeometry(QRect(10, 640, 500, 250));
        StateWidget->setFont(font);
        StateWidget->setStyleSheet(QStringLiteral("background:rgba(21, 79, 255, 150)"));
        OtherSideResult = new QLabel(StateWidget);
        OtherSideResult->setObjectName(QStringLiteral("OtherSideResult"));
        OtherSideResult->setEnabled(true);
        OtherSideResult->setGeometry(QRect(0, 0, 250, 250));
        OtherSideResult->setStyleSheet(QStringLiteral(""));
        OtherSideResult->setScaledContents(true);
        NetworkResult_OtherSide = new QLabel(StateWidget);
        NetworkResult_OtherSide->setObjectName(QStringLiteral("NetworkResult_OtherSide"));
        NetworkResult_OtherSide->setEnabled(true);
        NetworkResult_OtherSide->setGeometry(QRect(250, 0, 248, 250));
        NetworkResult_OtherSide->setStyleSheet(QStringLiteral(""));
        NetworkResult_OtherSide->setScaledContents(true);
        RenderGroupBox = new QGroupBox(centralWidget);
        RenderGroupBox->setObjectName(QStringLiteral("RenderGroupBox"));
        RenderGroupBox->setGeometry(QRect(1490, 10, 101, 181));
        OCTViewDir = new QComboBox(RenderGroupBox);
        OCTViewDir->addItem(QString());
        OCTViewDir->setObjectName(QStringLiteral("OCTViewDir"));
        OCTViewDir->setGeometry(QRect(10, 40, 81, 22));
        OCTViewDirText = new QLabel(RenderGroupBox);
        OCTViewDirText->setObjectName(QStringLiteral("OCTViewDirText"));
        OCTViewDirText->setGeometry(QRect(10, 20, 51, 16));
        DentistProjectV2Class->setCentralWidget(centralWidget);

        retranslateUi(DentistProjectV2Class);

        tabWidget->setCurrentIndex(2);
        ResetRotationMode->setDefault(false);
        GyroscopeResetToZero->setDefault(false);


        QMetaObject::connectSlotsByName(DentistProjectV2Class);
    } // setupUi

    void retranslateUi(QMainWindow *DentistProjectV2Class)
    {
        DentistProjectV2Class->setWindowTitle(QApplication::translate("DentistProjectV2Class", "DentistProjectV2", nullptr));
        ScanResult->setTitle(QApplication::translate("DentistProjectV2Class", "\346\216\203\346\217\217\347\265\220\346\236\234", nullptr));
        ImageResult->setText(QString());
        BorderDetectionResult->setText(QString());
        ImageResultText->setText(QApplication::translate("DentistProjectV2Class", "OCT \350\275\211\345\256\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
        BorderDetectionResultText->setText(QApplication::translate("DentistProjectV2Class", "\350\231\225\347\220\206\345\256\214 & \346\212\223\345\207\272\351\202\212\347\225\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
        ScanNum_Min->setText(QApplication::translate("DentistProjectV2Class", "60", nullptr));
        ScanNum_Max->setText(QApplication::translate("DentistProjectV2Class", "200", nullptr));
        ScanNum_Value->setText(QApplication::translate("DentistProjectV2Class", "60", nullptr));
        NetworkResultText->setText(QApplication::translate("DentistProjectV2Class", "\347\266\262\350\267\257\345\210\244\346\226\267\345\256\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
        NetworkResult->setText(QString());
        BLEDeviceBox->setTitle(QApplication::translate("DentistProjectV2Class", "\350\243\235\347\275\256\350\250\255\345\256\232", nullptr));
        BtnConnectCOM->setText(QApplication::translate("DentistProjectV2Class", "\351\200\243\347\265\220 COM Port", nullptr));
        BtnScanBLEDevice->setText(QApplication::translate("DentistProjectV2Class", "\346\220\234\345\260\213\350\227\215\350\212\275\351\200\243\347\267\232", nullptr));
        BtnConnectBLEDevice->setText(QApplication::translate("DentistProjectV2Class", "\345\273\272\347\253\213\350\227\215\350\212\275\351\200\243\347\267\232", nullptr));
        BtnSearchCom->setText(QApplication::translate("DentistProjectV2Class", "\346\220\234\345\260\213 COM Port", nullptr));
        BLEDeviceInfoBox->setTitle(QApplication::translate("DentistProjectV2Class", "\350\227\215\350\212\275\350\263\207\350\250\212", nullptr));
        EularText->setText(QApplication::translate("DentistProjectV2Class", "\357\274\270\357\274\232 0\n"
"\357\274\271\357\274\232 0\n"
"\357\274\272\357\274\232 0", nullptr));
        BLEStatus->setText(QApplication::translate("DentistProjectV2Class", "\350\227\215\350\212\275\347\213\200\346\205\213\357\274\232\346\234\252\351\200\243\346\216\245", nullptr));
        ResetRotationBox->setTitle(QApplication::translate("DentistProjectV2Class", "Rotation \350\250\255\345\256\232\347\233\270\351\227\234", nullptr));
        ResetRotationMode->setText(QApplication::translate("DentistProjectV2Class", "Rotation Mode (OFF)", nullptr));
        GyroscopeResetToZero->setText(QApplication::translate("DentistProjectV2Class", "\344\271\235\350\273\270\346\255\270\351\233\266", nullptr));
        BLETestingBox->setTitle(QApplication::translate("DentistProjectV2Class", "\350\227\215\350\212\275\346\270\254\350\251\246\347\233\270\351\227\234(\351\200\262\351\232\216)", nullptr));
        PointCloudAlignmentTestBtn->setText(QApplication::translate("DentistProjectV2Class", "\344\271\235\350\273\270\351\273\236\351\233\262\346\213\274\346\216\245\346\270\254\350\251\246", nullptr));
        BLEConnect_OneBtn->setText(QApplication::translate("DentistProjectV2Class", "\344\270\200    \351\215\265    \350\227\215    \350\212\275    \351\200\243    \347\267\232", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_Deivce), QApplication::translate("DentistProjectV2Class", "\350\227\215\350\212\275\350\243\235\347\275\256\350\250\255\345\256\232", nullptr));
        OCTNormalSettingBox->setTitle(QApplication::translate("DentistProjectV2Class", "\345\270\270\347\224\250\350\250\255\345\256\232", nullptr));
        SaveLocationLabel->setText(QApplication::translate("DentistProjectV2Class", "\345\204\262\345\255\230\350\263\207\346\226\231\347\232\204\350\267\257\345\276\221\357\274\232", nullptr));
        SaveLocationBtn->setText(QApplication::translate("DentistProjectV2Class", "\351\201\270\346\223\207\350\267\257\345\276\221", nullptr));
        AutoSaveSingleRawDataWhileScan_CheckBox->setText(QApplication::translate("DentistProjectV2Class", "\346\216\203\346\217\217\346\231\202\350\207\252\345\213\225\345\204\262\345\255\230\345\226\256\345\274\265 Raw Data", nullptr));
        ScanButton->setText(QApplication::translate("DentistProjectV2Class", "\346\216\203    \346\217\217    \346\250\241    \345\274\217\n"
"(End)", nullptr));
        AutoSaveImageWhileScan_CheckBox->setText(QApplication::translate("DentistProjectV2Class", "\346\216\203\346\217\217\346\231\202\350\207\252\345\213\225\350\275\211\346\210\220\345\275\261\345\203\217\344\270\246\345\204\262\345\255\230\347\265\220\346\236\234", nullptr));
        AutoSaveMultiRawDataWhileScan_CheckBox->setText(QApplication::translate("DentistProjectV2Class", "\346\216\203\346\217\217\346\231\202\350\207\252\345\213\225\345\204\262\345\255\230\347\253\213\351\253\224 Raw Data", nullptr));
        AutoDeleteShakeData_CheckBox->setText(QApplication::translate("DentistProjectV2Class", "\350\207\252\345\213\225\345\210\252\351\231\244\346\231\203\345\213\225\350\263\207\346\226\231", nullptr));
        ScanOnceButton->setText(QApplication::translate("DentistProjectV2Class", "\345\217\252\346\216\203\346\217\217\344\270\200\345\274\265\n"
"\346\216\203\345\210\260\344\270\215\345\213\225\347\202\272\346\255\242", nullptr));
        OCTTestingBox->setTitle(QApplication::translate("DentistProjectV2Class", "OCT \346\270\254\350\251\246\347\233\270\351\227\234 (\351\200\262\351\232\216)", nullptr));
        RawDataToImage->setText(QApplication::translate("DentistProjectV2Class", "\350\275\211\346\210\220\345\234\226\350\274\270\345\207\272", nullptr));
        EasyBorderDetect->setText(QApplication::translate("DentistProjectV2Class", "\347\260\241\346\230\223\351\202\212\347\225\214\346\270\254\350\251\246", nullptr));
        SingleImageShakeTestButton->setText(QApplication::translate("DentistProjectV2Class", "\345\226\256\345\274\265\346\231\203\345\213\225\345\201\265\346\270\254", nullptr));
        SegNetTestButton->setText(QApplication::translate("DentistProjectV2Class", "SegNet \351\240\220\346\270\254", nullptr));
        BeepSoundTestButton->setText(QApplication::translate("DentistProjectV2Class", "Beep Sound \346\270\254\350\251\246", nullptr));
        MultiImageShakeTestButton->setText(QApplication::translate("DentistProjectV2Class", "\345\244\232\345\274\265\346\231\203\345\213\225\345\201\265\346\270\254", nullptr));
        SlimLabViewRawData->setText(QApplication::translate("DentistProjectV2Class", "\347\270\256\346\270\233 Labview Data", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_OCT), QApplication::translate("DentistProjectV2Class", "OCT \350\243\235\347\275\256\350\250\255\345\256\232", nullptr));
        AlignBox->setTitle(QApplication::translate("DentistProjectV2Class", "Align \347\233\270\351\227\234", nullptr));
        AlignLastTwoPCButton->setText(QApplication::translate("DentistProjectV2Class", "\346\213\274\346\216\245\345\276\214\351\235\242\345\205\251\347\211\207\351\273\236\351\233\262", nullptr));
        CombineLastTwoPCButton->setText(QApplication::translate("DentistProjectV2Class", "\345\220\210\344\275\265\346\234\200\345\276\214\345\205\251\347\211\207\351\273\236\351\233\262", nullptr));
        CombineAllPCButton->setText(QApplication::translate("DentistProjectV2Class", "\345\220\210\344\275\265\346\211\200\346\234\211\351\273\236\351\233\262", nullptr));
        AlignmentAllPCTest->setText(QApplication::translate("DentistProjectV2Class", "\344\270\200\347\263\273\345\210\227 PointCloud\n"
"\346\213\274\346\216\245\346\270\254\350\251\246", nullptr));
        PassScanDataToPC->setText(QApplication::translate("DentistProjectV2Class", "\344\270\200\347\263\273\345\210\227 RawData\n"
"\346\216\203\346\217\217\345\234\226\350\275\211\351\273\236\351\233\262", nullptr));
        AveragePCErrorTest->setText(QApplication::translate("DentistProjectV2Class", "\345\271\263\345\235\207\345\244\232\347\211\207\351\273\236\351\233\262\350\252\244\345\267\256\n"
"(\346\270\254\350\251\246\347\224\250)", nullptr));
        PCInfoBox->setTitle(QApplication::translate("DentistProjectV2Class", "\351\273\236\351\233\262\350\263\207\350\250\212", nullptr));
        ChoosePCIndexText->setText(QApplication::translate("DentistProjectV2Class", "\351\273\236\351\233\262Index", nullptr));
        PCOperationBox->setTitle(QApplication::translate("DentistProjectV2Class", "\351\273\236\351\233\262\345\204\262\345\255\230\345\222\214\350\256\200\345\217\226", nullptr));
        LoadPCButton->setText(QApplication::translate("DentistProjectV2Class", "\350\256\200\345\217\226\351\273\236\351\233\262", nullptr));
        SavePCButton->setText(QApplication::translate("DentistProjectV2Class", "\345\204\262\345\255\230\351\273\236\351\233\262", nullptr));
        DeletePCButton->setText(QApplication::translate("DentistProjectV2Class", "\345\210\252\351\231\244\351\273\236\351\233\262", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_PC), QApplication::translate("DentistProjectV2Class", "\351\273\236\351\233\262\346\223\215\344\275\234", nullptr));
        NetworkDataOperationBox->setTitle(QApplication::translate("DentistProjectV2Class", "\347\224\242\347\224\237\350\263\207\346\226\231\347\233\270\351\227\234", nullptr));
        DataGenerationBtn->setText(QApplication::translate("DentistProjectV2Class", "\347\224\242\347\224\237Trainning\347\232\204\350\263\207\346\226\231", nullptr));
        NetworkDataTestBox->setTitle(QApplication::translate("DentistProjectV2Class", "\351\241\236\347\245\236\347\266\223\347\266\262\350\267\257\350\263\207\346\226\231\346\270\254\350\251\246\347\233\270\351\227\234", nullptr));
        VolumeRenderingTestBtn->setText(QApplication::translate("DentistProjectV2Class", "Rendering\346\270\254\350\251\246", nullptr));
        PredictResultTestingBtn->setText(QApplication::translate("DentistProjectV2Class", "\351\240\220\346\270\254\347\265\220\346\236\234", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_Network), QApplication::translate("DentistProjectV2Class", "\347\266\262\350\267\257\347\233\270\351\227\234", nullptr));
        OtherSideResult->setText(QString());
        NetworkResult_OtherSide->setText(QString());
        RenderGroupBox->setTitle(QApplication::translate("DentistProjectV2Class", "Render\351\201\270\351\240\205", nullptr));
        OCTViewDir->setItemText(0, QApplication::translate("DentistProjectV2Class", "\347\224\261\344\270\212\345\276\200\344\270\213", nullptr));

        OCTViewDirText->setText(QApplication::translate("DentistProjectV2Class", "OCT\350\246\226\350\247\222", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistProjectV2Class: public Ui_DentistProjectV2Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTPROJECTV2_H
