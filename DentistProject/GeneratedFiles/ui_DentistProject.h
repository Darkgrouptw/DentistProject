/********************************************************************************
** Form generated from reading UI file 'DentistProject.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DENTISTPROJECT_H
#define UI_DENTISTPROJECT_H

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
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>
#include "openglwidget.h"

QT_BEGIN_NAMESPACE

class Ui_DentistProjectClass
{
public:
    QAction *actionLoadSTL;
    QWidget *centralWidget;
    OpenGLWidget *DisplayPanel;
    QGroupBox *RenderOptionsBox;
    QCheckBox *RenderBorder_CheckBox;
    QCheckBox *RenderTriangle_CheckBox;
    QCheckBox *RenderPointDot_CheckBox;
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
    QLabel *QuaternionText;
    QLabel *BLEStatus;
    QWidget *Tab_OCT;
    QGroupBox *OCTNormalSettingBox;
    QLineEdit *SaveLocationText;
    QLabel *SaveLocationLabel;
    QPushButton *SaveLocationBtn;
    QCheckBox *SaveWithTime_CheckBox;
    QCheckBox *AutoScanRawDataWhileScan_CheckBox;
    QPushButton *ScanButton;
    QCheckBox *AutoScanImageWhileScan_CheckBox;
    QGroupBox *OCTTestingBox;
    QPushButton *RawDataToImage;
    QPushButton *EasyBorderDetect;
    QPushButton *RawDataCheck;
    QPushButton *ShakeTestButton;
    QPushButton *SegNetTestButton;
    QPushButton *BeepSoundTestButton;
    QGroupBox *ScanResult;
    QLabel *ImageResult;
    QLabel *FinalResult;
    QLabel *ImageResultText;
    QLabel *FinalResultText;
    QSlider *ScanNumSlider;
    QLabel *ScanNum_Min;
    QLabel *ScanNum_Max;
    QLabel *ScanNum_Value;
    QLabel *NetworkResult;
    QLabel *NetworkResultText;
    QWidget *StateWidget;
    QProgressBar *progressBar;
    QToolBar *mainToolBar;

    void setupUi(QMainWindow *DentistProjectClass)
    {
        if (DentistProjectClass->objectName().isEmpty())
            DentistProjectClass->setObjectName(QStringLiteral("DentistProjectClass"));
        DentistProjectClass->resize(1600, 900);
        actionLoadSTL = new QAction(DentistProjectClass);
        actionLoadSTL->setObjectName(QStringLiteral("actionLoadSTL"));
        centralWidget = new QWidget(DentistProjectClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        DisplayPanel = new OpenGLWidget(centralWidget);
        DisplayPanel->setObjectName(QStringLiteral("DisplayPanel"));
        DisplayPanel->setGeometry(QRect(0, 0, 900, 900));
        RenderOptionsBox = new QGroupBox(centralWidget);
        RenderOptionsBox->setObjectName(QStringLiteral("RenderOptionsBox"));
        RenderOptionsBox->setGeometry(QRect(1500, 10, 90, 211));
        RenderBorder_CheckBox = new QCheckBox(RenderOptionsBox);
        RenderBorder_CheckBox->setObjectName(QStringLiteral("RenderBorder_CheckBox"));
        RenderBorder_CheckBox->setGeometry(QRect(10, 60, 61, 21));
        RenderBorder_CheckBox->setChecked(true);
        RenderTriangle_CheckBox = new QCheckBox(RenderOptionsBox);
        RenderTriangle_CheckBox->setObjectName(QStringLiteral("RenderTriangle_CheckBox"));
        RenderTriangle_CheckBox->setGeometry(QRect(10, 30, 61, 21));
        RenderTriangle_CheckBox->setChecked(true);
        RenderPointDot_CheckBox = new QCheckBox(RenderOptionsBox);
        RenderPointDot_CheckBox->setObjectName(QStringLiteral("RenderPointDot_CheckBox"));
        RenderPointDot_CheckBox->setGeometry(QRect(10, 90, 61, 21));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(900, 570, 710, 320));
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
        BLEDeviceBox->setGeometry(QRect(10, 10, 511, 101));
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
        BLEDeviceInfoBox->setGeometry(QRect(20, 110, 181, 161));
        QuaternionText = new QLabel(BLEDeviceInfoBox);
        QuaternionText->setObjectName(QStringLiteral("QuaternionText"));
        QuaternionText->setGeometry(QRect(10, 50, 121, 101));
        QFont font;
        font.setPointSize(12);
        QuaternionText->setFont(font);
        BLEStatus = new QLabel(BLEDeviceInfoBox);
        BLEStatus->setObjectName(QStringLiteral("BLEStatus"));
        BLEStatus->setGeometry(QRect(10, 20, 161, 31));
        BLEStatus->setFont(font);
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
        SaveWithTime_CheckBox = new QCheckBox(OCTNormalSettingBox);
        SaveWithTime_CheckBox->setObjectName(QStringLiteral("SaveWithTime_CheckBox"));
        SaveWithTime_CheckBox->setGeometry(QRect(10, 80, 91, 16));
        SaveWithTime_CheckBox->setChecked(true);
        AutoScanRawDataWhileScan_CheckBox = new QCheckBox(OCTNormalSettingBox);
        AutoScanRawDataWhileScan_CheckBox->setObjectName(QStringLiteral("AutoScanRawDataWhileScan_CheckBox"));
        AutoScanRawDataWhileScan_CheckBox->setGeometry(QRect(10, 110, 161, 16));
        AutoScanRawDataWhileScan_CheckBox->setChecked(true);
        ScanButton = new QPushButton(OCTNormalSettingBox);
        ScanButton->setObjectName(QStringLiteral("ScanButton"));
        ScanButton->setGeometry(QRect(360, 130, 151, 101));
        AutoScanImageWhileScan_CheckBox = new QCheckBox(OCTNormalSettingBox);
        AutoScanImageWhileScan_CheckBox->setObjectName(QStringLiteral("AutoScanImageWhileScan_CheckBox"));
        AutoScanImageWhileScan_CheckBox->setGeometry(QRect(10, 140, 151, 16));
        AutoScanImageWhileScan_CheckBox->setChecked(false);
        OCTTestingBox = new QGroupBox(Tab_OCT);
        OCTTestingBox->setObjectName(QStringLiteral("OCTTestingBox"));
        OCTTestingBox->setGeometry(QRect(540, 10, 151, 271));
        RawDataToImage = new QPushButton(OCTTestingBox);
        RawDataToImage->setObjectName(QStringLiteral("RawDataToImage"));
        RawDataToImage->setGeometry(QRect(10, 20, 131, 23));
        EasyBorderDetect = new QPushButton(OCTTestingBox);
        EasyBorderDetect->setObjectName(QStringLiteral("EasyBorderDetect"));
        EasyBorderDetect->setGeometry(QRect(10, 80, 131, 23));
        RawDataCheck = new QPushButton(OCTTestingBox);
        RawDataCheck->setObjectName(QStringLiteral("RawDataCheck"));
        RawDataCheck->setGeometry(QRect(10, 50, 131, 23));
        ShakeTestButton = new QPushButton(OCTTestingBox);
        ShakeTestButton->setObjectName(QStringLiteral("ShakeTestButton"));
        ShakeTestButton->setGeometry(QRect(10, 110, 131, 23));
        SegNetTestButton = new QPushButton(OCTTestingBox);
        SegNetTestButton->setObjectName(QStringLiteral("SegNetTestButton"));
        SegNetTestButton->setGeometry(QRect(10, 240, 131, 23));
        BeepSoundTestButton = new QPushButton(OCTTestingBox);
        BeepSoundTestButton->setObjectName(QStringLiteral("BeepSoundTestButton"));
        BeepSoundTestButton->setGeometry(QRect(10, 210, 131, 23));
        tabWidget->addTab(Tab_OCT, QString());
        ScanResult = new QGroupBox(centralWidget);
        ScanResult->setObjectName(QStringLiteral("ScanResult"));
        ScanResult->setGeometry(QRect(910, 0, 581, 561));
        ImageResult = new QLabel(ScanResult);
        ImageResult->setObjectName(QStringLiteral("ImageResult"));
        ImageResult->setGeometry(QRect(10, 40, 550, 135));
        ImageResult->setStyleSheet(QStringLiteral(""));
        ImageResult->setScaledContents(true);
        FinalResult = new QLabel(ScanResult);
        FinalResult->setObjectName(QStringLiteral("FinalResult"));
        FinalResult->setGeometry(QRect(10, 360, 550, 135));
        FinalResult->setStyleSheet(QStringLiteral(""));
        FinalResult->setScaledContents(true);
        ImageResultText = new QLabel(ScanResult);
        ImageResultText->setObjectName(QStringLiteral("ImageResultText"));
        ImageResultText->setGeometry(QRect(10, 20, 101, 16));
        FinalResultText = new QLabel(ScanResult);
        FinalResultText->setObjectName(QStringLiteral("FinalResultText"));
        FinalResultText->setGeometry(QRect(10, 340, 151, 16));
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
        NetworkResult = new QLabel(ScanResult);
        NetworkResult->setObjectName(QStringLiteral("NetworkResult"));
        NetworkResult->setGeometry(QRect(10, 200, 550, 135));
        NetworkResult->setStyleSheet(QStringLiteral(""));
        NetworkResult->setScaledContents(true);
        NetworkResultText = new QLabel(ScanResult);
        NetworkResultText->setObjectName(QStringLiteral("NetworkResultText"));
        NetworkResultText->setGeometry(QRect(10, 180, 111, 16));
        StateWidget = new QWidget(centralWidget);
        StateWidget->setObjectName(QStringLiteral("StateWidget"));
        StateWidget->setGeometry(QRect(0, 750, 341, 131));
        StateWidget->setStyleSheet(QStringLiteral("background:rgba(21, 79, 255, 150)"));
        progressBar = new QProgressBar(StateWidget);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setGeometry(QRect(0, 100, 341, 21));
        QFont font1;
        font1.setFamily(QStringLiteral("Adobe Arabic"));
        font1.setPointSize(24);
        progressBar->setFont(font1);
        progressBar->setStyleSheet(QStringLiteral("color : white;"));
        progressBar->setValue(24);
        DentistProjectClass->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(DentistProjectClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        DentistProjectClass->addToolBar(Qt::TopToolBarArea, mainToolBar);

        mainToolBar->addAction(actionLoadSTL);

        retranslateUi(DentistProjectClass);

        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(DentistProjectClass);
    } // setupUi

    void retranslateUi(QMainWindow *DentistProjectClass)
    {
        DentistProjectClass->setWindowTitle(QApplication::translate("DentistProjectClass", "DentistProject", nullptr));
        actionLoadSTL->setText(QApplication::translate("DentistProjectClass", "1. \350\256\200\345\217\226 STL(From 3Shape)", nullptr));
        RenderOptionsBox->setTitle(QApplication::translate("DentistProjectClass", "Render\351\201\270\351\240\205", nullptr));
        RenderBorder_CheckBox->setText(QApplication::translate("DentistProjectClass", "\347\225\253\351\202\212\347\225\214", nullptr));
        RenderTriangle_CheckBox->setText(QApplication::translate("DentistProjectClass", "\347\225\253\351\235\242", nullptr));
        RenderPointDot_CheckBox->setText(QApplication::translate("DentistProjectClass", "\347\225\253\351\273\236\351\233\262", nullptr));
        BLEDeviceBox->setTitle(QApplication::translate("DentistProjectClass", "\350\243\235\347\275\256\350\250\255\345\256\232", nullptr));
        BtnConnectCOM->setText(QApplication::translate("DentistProjectClass", "\351\200\243\347\265\220 COM Port", nullptr));
        BtnScanBLEDevice->setText(QApplication::translate("DentistProjectClass", "\346\220\234\345\260\213\350\227\215\350\212\275\351\200\243\347\267\232", nullptr));
        BtnConnectBLEDevice->setText(QApplication::translate("DentistProjectClass", "\345\273\272\347\253\213\350\227\215\350\212\275\351\200\243\347\267\232", nullptr));
        BtnSearchCom->setText(QApplication::translate("DentistProjectClass", "\346\220\234\345\260\213 COM Port", nullptr));
        BLEDeviceInfoBox->setTitle(QApplication::translate("DentistProjectClass", "\350\227\215\350\212\275\350\263\207\350\250\212", nullptr));
        QuaternionText->setText(QApplication::translate("DentistProjectClass", "\357\274\267\357\274\232 0\n"
"\357\274\270\357\274\232 0\n"
"\357\274\271\357\274\232 0\n"
"\357\274\272\357\274\232 0", nullptr));
        BLEStatus->setText(QApplication::translate("DentistProjectClass", "\350\227\215\350\212\275\347\213\200\346\205\213\357\274\232\346\234\252\351\200\243\346\216\245", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_Deivce), QApplication::translate("DentistProjectClass", "\350\227\215\350\212\275\350\243\235\347\275\256", nullptr));
        OCTNormalSettingBox->setTitle(QApplication::translate("DentistProjectClass", "\345\270\270\347\224\250\350\250\255\345\256\232", nullptr));
        SaveLocationLabel->setText(QApplication::translate("DentistProjectClass", "\345\204\262\345\255\230\350\263\207\346\226\231\347\232\204\350\267\257\345\276\221\357\274\232", nullptr));
        SaveLocationBtn->setText(QApplication::translate("DentistProjectClass", "\351\201\270\346\223\207\350\267\257\345\276\221", nullptr));
        SaveWithTime_CheckBox->setText(QApplication::translate("DentistProjectClass", "\344\273\245\346\231\202\351\226\223\345\204\262\345\255\230", nullptr));
        AutoScanRawDataWhileScan_CheckBox->setText(QApplication::translate("DentistProjectClass", "\346\216\203\346\217\217\346\231\202\350\207\252\345\213\225\345\204\262\345\255\230 Raw Data", nullptr));
        ScanButton->setText(QApplication::translate("DentistProjectClass", "\346\216\203    \346\217\217", nullptr));
        AutoScanImageWhileScan_CheckBox->setText(QApplication::translate("DentistProjectClass", "\346\216\203\346\217\217\346\231\202\350\207\252\345\213\225\345\204\262\345\255\230\345\275\261\345\203\217\347\265\220\346\236\234", nullptr));
        OCTTestingBox->setTitle(QApplication::translate("DentistProjectClass", "OCT \346\270\254\350\251\246\347\233\270\351\227\234 (\351\200\262\351\232\216)", nullptr));
        RawDataToImage->setText(QApplication::translate("DentistProjectClass", "\350\274\270\345\207\272\345\234\226 (Raw Data)", nullptr));
        EasyBorderDetect->setText(QApplication::translate("DentistProjectClass", "\347\260\241\346\230\223\351\202\212\347\225\214\346\270\254\350\251\246", nullptr));
        RawDataCheck->setText(QApplication::translate("DentistProjectClass", "\350\274\270\345\207\272\350\263\207\346\226\231 (Raw Data)", nullptr));
        ShakeTestButton->setText(QApplication::translate("DentistProjectClass", "\346\231\203\345\213\225\345\201\265\346\270\254", nullptr));
        SegNetTestButton->setText(QApplication::translate("DentistProjectClass", "SegNet \351\240\220\346\270\254", nullptr));
        BeepSoundTestButton->setText(QApplication::translate("DentistProjectClass", "Beep Sound \346\270\254\350\251\246", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_OCT), QApplication::translate("DentistProjectClass", "OCT \350\243\235\347\275\256\350\250\255\345\256\232", nullptr));
        ScanResult->setTitle(QApplication::translate("DentistProjectClass", "\346\216\203\346\217\217\347\265\220\346\236\234", nullptr));
        ImageResult->setText(QString());
        FinalResult->setText(QString());
        ImageResultText->setText(QApplication::translate("DentistProjectClass", "OCT \350\275\211\345\256\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
        FinalResultText->setText(QApplication::translate("DentistProjectClass", "\350\231\225\347\220\206\345\256\214 & \346\212\223\345\207\272\351\202\212\347\225\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
        ScanNum_Min->setText(QApplication::translate("DentistProjectClass", "60", nullptr));
        ScanNum_Max->setText(QApplication::translate("DentistProjectClass", "200", nullptr));
        ScanNum_Value->setText(QApplication::translate("DentistProjectClass", "60", nullptr));
        NetworkResult->setText(QString());
        NetworkResultText->setText(QApplication::translate("DentistProjectClass", "\347\266\262\350\267\257\345\210\244\346\226\267\345\256\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistProjectClass: public Ui_DentistProjectClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTPROJECT_H
