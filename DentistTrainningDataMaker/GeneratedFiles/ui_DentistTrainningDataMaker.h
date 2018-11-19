/********************************************************************************
** Form generated from reading UI file 'DentistTrainningDataMaker.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DENTISTTRAINNINGDATAMAKER_H
#define UI_DENTISTTRAINNINGDATAMAKER_H

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
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>
#include "openglwidget.h"

QT_BEGIN_NAMESPACE

class Ui_DentistTrainningDataMakerClass
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
    QPushButton *RawDataToImage;
    QPushButton *EasyBorderDetect;
    QGroupBox *OCTNormalSettingbOX;
    QLineEdit *SaveLocationText;
    QLabel *SaveLocationLabel;
    QPushButton *SaveLocationBtn;
    QCheckBox *SaveWithTime_CheckBox;
    QCheckBox *AutoScanWhileScan_CheckBox;
    QPushButton *ScanButton;
    QPushButton *BeepSoundTest;
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
    QToolBar *mainToolBar;

    void setupUi(QMainWindow *DentistTrainningDataMakerClass)
    {
        if (DentistTrainningDataMakerClass->objectName().isEmpty())
            DentistTrainningDataMakerClass->setObjectName(QStringLiteral("DentistTrainningDataMakerClass"));
        DentistTrainningDataMakerClass->resize(1600, 900);
        actionLoadSTL = new QAction(DentistTrainningDataMakerClass);
        actionLoadSTL->setObjectName(QStringLiteral("actionLoadSTL"));
        centralWidget = new QWidget(DentistTrainningDataMakerClass);
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
        tabWidget->setGeometry(QRect(900, 570, 700, 300));
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
        RawDataToImage = new QPushButton(Tab_OCT);
        RawDataToImage->setObjectName(QStringLiteral("RawDataToImage"));
        RawDataToImage->setGeometry(QRect(560, 50, 131, 23));
        EasyBorderDetect = new QPushButton(Tab_OCT);
        EasyBorderDetect->setObjectName(QStringLiteral("EasyBorderDetect"));
        EasyBorderDetect->setGeometry(QRect(560, 80, 131, 23));
        OCTNormalSettingbOX = new QGroupBox(Tab_OCT);
        OCTNormalSettingbOX->setObjectName(QStringLiteral("OCTNormalSettingbOX"));
        OCTNormalSettingbOX->setGeometry(QRect(10, 10, 541, 251));
        SaveLocationText = new QLineEdit(OCTNormalSettingbOX);
        SaveLocationText->setObjectName(QStringLiteral("SaveLocationText"));
        SaveLocationText->setEnabled(false);
        SaveLocationText->setGeometry(QRect(10, 40, 431, 20));
        SaveLocationLabel = new QLabel(OCTNormalSettingbOX);
        SaveLocationLabel->setObjectName(QStringLiteral("SaveLocationLabel"));
        SaveLocationLabel->setGeometry(QRect(10, 20, 111, 16));
        SaveLocationBtn = new QPushButton(OCTNormalSettingbOX);
        SaveLocationBtn->setObjectName(QStringLiteral("SaveLocationBtn"));
        SaveLocationBtn->setGeometry(QRect(450, 40, 75, 23));
        SaveWithTime_CheckBox = new QCheckBox(OCTNormalSettingbOX);
        SaveWithTime_CheckBox->setObjectName(QStringLiteral("SaveWithTime_CheckBox"));
        SaveWithTime_CheckBox->setGeometry(QRect(10, 80, 91, 16));
        SaveWithTime_CheckBox->setChecked(true);
        AutoScanWhileScan_CheckBox = new QCheckBox(OCTNormalSettingbOX);
        AutoScanWhileScan_CheckBox->setObjectName(QStringLiteral("AutoScanWhileScan_CheckBox"));
        AutoScanWhileScan_CheckBox->setGeometry(QRect(10, 110, 111, 16));
        AutoScanWhileScan_CheckBox->setChecked(true);
        ScanButton = new QPushButton(OCTNormalSettingbOX);
        ScanButton->setObjectName(QStringLiteral("ScanButton"));
        ScanButton->setGeometry(QRect(370, 130, 151, 101));
        BeepSoundTest = new QPushButton(Tab_OCT);
        BeepSoundTest->setObjectName(QStringLiteral("BeepSoundTest"));
        BeepSoundTest->setGeometry(QRect(560, 110, 131, 23));
        tabWidget->addTab(Tab_OCT, QString());
        ScanResult = new QGroupBox(centralWidget);
        ScanResult->setObjectName(QStringLiteral("ScanResult"));
        ScanResult->setGeometry(QRect(910, 0, 581, 561));
        ImageResult = new QLabel(ScanResult);
        ImageResult->setObjectName(QStringLiteral("ImageResult"));
        ImageResult->setGeometry(QRect(10, 40, 550, 135));
        ImageResult->setStyleSheet(QStringLiteral(""));
        FinalResult = new QLabel(ScanResult);
        FinalResult->setObjectName(QStringLiteral("FinalResult"));
        FinalResult->setGeometry(QRect(10, 360, 550, 135));
        FinalResult->setStyleSheet(QStringLiteral(""));
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
        ScanNumSlider->setMinimum(0);
        ScanNumSlider->setMaximum(249);
        ScanNumSlider->setValue(0);
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
        NetworkResultText = new QLabel(ScanResult);
        NetworkResultText->setObjectName(QStringLiteral("NetworkResultText"));
        NetworkResultText->setGeometry(QRect(10, 180, 111, 16));
        DentistTrainningDataMakerClass->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(DentistTrainningDataMakerClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        DentistTrainningDataMakerClass->addToolBar(Qt::TopToolBarArea, mainToolBar);

        mainToolBar->addAction(actionLoadSTL);

        retranslateUi(DentistTrainningDataMakerClass);

        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(DentistTrainningDataMakerClass);
    } // setupUi

    void retranslateUi(QMainWindow *DentistTrainningDataMakerClass)
    {
        DentistTrainningDataMakerClass->setWindowTitle(QApplication::translate("DentistTrainningDataMakerClass", "DentistTrainningDataMaker", nullptr));
        actionLoadSTL->setText(QApplication::translate("DentistTrainningDataMakerClass", "1. \350\256\200\345\217\226 STL(From 3Shape)", nullptr));
        RenderOptionsBox->setTitle(QApplication::translate("DentistTrainningDataMakerClass", "Render\351\201\270\351\240\205", nullptr));
        RenderBorder_CheckBox->setText(QApplication::translate("DentistTrainningDataMakerClass", "\347\225\253\351\202\212\347\225\214", nullptr));
        RenderTriangle_CheckBox->setText(QApplication::translate("DentistTrainningDataMakerClass", "\347\225\253\351\235\242", nullptr));
        RenderPointDot_CheckBox->setText(QApplication::translate("DentistTrainningDataMakerClass", "\347\225\253\351\273\236\351\233\262", nullptr));
        BLEDeviceBox->setTitle(QApplication::translate("DentistTrainningDataMakerClass", "\350\243\235\347\275\256\350\250\255\345\256\232", nullptr));
        BtnConnectCOM->setText(QApplication::translate("DentistTrainningDataMakerClass", "\351\200\243\347\265\220 COM Port", nullptr));
        BtnScanBLEDevice->setText(QApplication::translate("DentistTrainningDataMakerClass", "\346\220\234\345\260\213\350\227\215\350\212\275\351\200\243\347\267\232", nullptr));
        BtnConnectBLEDevice->setText(QApplication::translate("DentistTrainningDataMakerClass", "\345\273\272\347\253\213\350\227\215\350\212\275\351\200\243\347\267\232", nullptr));
        BtnSearchCom->setText(QApplication::translate("DentistTrainningDataMakerClass", "\346\220\234\345\260\213 COM Port", nullptr));
        BLEDeviceInfoBox->setTitle(QApplication::translate("DentistTrainningDataMakerClass", "\350\227\215\350\212\275\350\263\207\350\250\212", nullptr));
        QuaternionText->setText(QApplication::translate("DentistTrainningDataMakerClass", "\357\274\267\357\274\232 0\n"
"\357\274\270\357\274\232 0\n"
"\357\274\271\357\274\232 0\n"
"\357\274\272\357\274\232 0", nullptr));
        BLEStatus->setText(QApplication::translate("DentistTrainningDataMakerClass", "\350\227\215\350\212\275\347\213\200\346\205\213\357\274\232\346\234\252\351\200\243\346\216\245", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_Deivce), QApplication::translate("DentistTrainningDataMakerClass", "\350\227\215\350\212\275\350\243\235\347\275\256", nullptr));
        RawDataToImage->setText(QApplication::translate("DentistTrainningDataMakerClass", "Raw Data \350\275\211\346\210\220\345\234\226\350\274\270\345\207\272", nullptr));
        EasyBorderDetect->setText(QApplication::translate("DentistTrainningDataMakerClass", "\347\260\241\346\230\223\351\202\212\347\225\214\346\270\254\350\251\246", nullptr));
        OCTNormalSettingbOX->setTitle(QApplication::translate("DentistTrainningDataMakerClass", "\345\270\270\347\224\250\350\250\255\345\256\232", nullptr));
        SaveLocationLabel->setText(QApplication::translate("DentistTrainningDataMakerClass", "\345\204\262\345\255\230\350\263\207\346\226\231\347\232\204\350\267\257\345\276\221\357\274\232", nullptr));
        SaveLocationBtn->setText(QApplication::translate("DentistTrainningDataMakerClass", "\351\201\270\346\223\207\350\267\257\345\276\221", nullptr));
        SaveWithTime_CheckBox->setText(QApplication::translate("DentistTrainningDataMakerClass", "\344\273\245\346\231\202\351\226\223\345\204\262\345\255\230", nullptr));
        AutoScanWhileScan_CheckBox->setText(QApplication::translate("DentistTrainningDataMakerClass", "\346\216\203\346\217\217\346\231\202\350\207\252\345\213\225\345\204\262\345\255\230", nullptr));
        ScanButton->setText(QApplication::translate("DentistTrainningDataMakerClass", "\346\216\203    \346\217\217", nullptr));
        BeepSoundTest->setText(QApplication::translate("DentistTrainningDataMakerClass", "Beep Sound \346\270\254\350\251\246", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_OCT), QApplication::translate("DentistTrainningDataMakerClass", "OCT \350\243\235\347\275\256\350\250\255\345\256\232", nullptr));
        ScanResult->setTitle(QApplication::translate("DentistTrainningDataMakerClass", "\346\216\203\346\217\217\347\265\220\346\236\234", nullptr));
        ImageResult->setText(QString());
        FinalResult->setText(QString());
        ImageResultText->setText(QApplication::translate("DentistTrainningDataMakerClass", "OCT \350\275\211\345\256\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
        FinalResultText->setText(QApplication::translate("DentistTrainningDataMakerClass", "\350\231\225\347\220\206\345\256\214 & \346\212\223\345\207\272\351\202\212\347\225\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
        ScanNum_Min->setText(QApplication::translate("DentistTrainningDataMakerClass", "0", nullptr));
        ScanNum_Max->setText(QApplication::translate("DentistTrainningDataMakerClass", "249", nullptr));
        ScanNum_Value->setText(QApplication::translate("DentistTrainningDataMakerClass", "0", nullptr));
        NetworkResult->setText(QString());
        NetworkResultText->setText(QApplication::translate("DentistTrainningDataMakerClass", "\347\266\262\350\267\257\345\210\244\346\226\267\345\256\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistTrainningDataMakerClass: public Ui_DentistTrainningDataMakerClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTTRAINNINGDATAMAKER_H
