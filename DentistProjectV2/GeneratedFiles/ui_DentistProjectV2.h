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
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DentistProjectV2Class
{
public:
    QWidget *centralWidget;
    QOpenGLWidget *openGLWidget;
    QGroupBox *ScanResult;
    QLabel *ImageResult;
    QLabel *FinalResult;
    QLabel *ImageResultText;
    QLabel *FinalResultText;
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

    void setupUi(QMainWindow *DentistProjectV2Class)
    {
        if (DentistProjectV2Class->objectName().isEmpty())
            DentistProjectV2Class->setObjectName(QStringLiteral("DentistProjectV2Class"));
        DentistProjectV2Class->resize(1600, 900);
        centralWidget = new QWidget(DentistProjectV2Class);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        openGLWidget = new QOpenGLWidget(centralWidget);
        openGLWidget->setObjectName(QStringLiteral("openGLWidget"));
        openGLWidget->setGeometry(QRect(0, 0, 900, 900));
        ScanResult = new QGroupBox(centralWidget);
        ScanResult->setObjectName(QStringLiteral("ScanResult"));
        ScanResult->setGeometry(QRect(900, 0, 581, 561));
        ImageResult = new QLabel(ScanResult);
        ImageResult->setObjectName(QStringLiteral("ImageResult"));
        ImageResult->setGeometry(QRect(10, 40, 550, 135));
        ImageResult->setStyleSheet(QStringLiteral(""));
        ImageResult->setScaledContents(true);
        FinalResult = new QLabel(ScanResult);
        FinalResult->setObjectName(QStringLiteral("FinalResult"));
        FinalResult->setGeometry(QRect(10, 200, 550, 135));
        FinalResult->setStyleSheet(QStringLiteral(""));
        FinalResult->setScaledContents(true);
        ImageResultText = new QLabel(ScanResult);
        ImageResultText->setObjectName(QStringLiteral("ImageResultText"));
        ImageResultText->setGeometry(QRect(10, 20, 101, 16));
        FinalResultText = new QLabel(ScanResult);
        FinalResultText->setObjectName(QStringLiteral("FinalResultText"));
        FinalResultText->setGeometry(QRect(10, 180, 151, 16));
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
        NetworkResultText->setEnabled(false);
        NetworkResultText->setGeometry(QRect(10, 340, 111, 16));
        NetworkResult = new QLabel(ScanResult);
        NetworkResult->setObjectName(QStringLiteral("NetworkResult"));
        NetworkResult->setEnabled(false);
        NetworkResult->setGeometry(QRect(10, 360, 550, 135));
        NetworkResult->setStyleSheet(QStringLiteral(""));
        NetworkResult->setScaledContents(true);
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
        ResetRotationBox->setGeometry(QRect(210, 130, 151, 141));
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
        BLETestingBox->setEnabled(false);
        BLETestingBox->setGeometry(QRect(540, 10, 151, 271));
        PointCloudAlignmentTestBtn = new QPushButton(BLETestingBox);
        PointCloudAlignmentTestBtn->setObjectName(QStringLiteral("PointCloudAlignmentTestBtn"));
        PointCloudAlignmentTestBtn->setGeometry(QRect(10, 20, 131, 23));
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
        OCTTestingBox->setEnabled(false);
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
        DentistProjectV2Class->setCentralWidget(centralWidget);

        retranslateUi(DentistProjectV2Class);

        tabWidget->setCurrentIndex(1);
        ResetRotationMode->setDefault(false);
        GyroscopeResetToZero->setDefault(false);


        QMetaObject::connectSlotsByName(DentistProjectV2Class);
    } // setupUi

    void retranslateUi(QMainWindow *DentistProjectV2Class)
    {
        DentistProjectV2Class->setWindowTitle(QApplication::translate("DentistProjectV2Class", "DentistProjectV2", nullptr));
        ScanResult->setTitle(QApplication::translate("DentistProjectV2Class", "\346\216\203\346\217\217\347\265\220\346\236\234", nullptr));
        ImageResult->setText(QString());
        FinalResult->setText(QString());
        ImageResultText->setText(QApplication::translate("DentistProjectV2Class", "OCT \350\275\211\345\256\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
        FinalResultText->setText(QApplication::translate("DentistProjectV2Class", "\350\231\225\347\220\206\345\256\214 & \346\212\223\345\207\272\351\202\212\347\225\214\347\232\204\347\265\220\346\236\234\357\274\232", nullptr));
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
        tabWidget->setTabText(tabWidget->indexOf(Tab_Deivce), QApplication::translate("DentistProjectV2Class", "\350\227\215\350\212\275\350\243\235\347\275\256", nullptr));
        OCTNormalSettingBox->setTitle(QApplication::translate("DentistProjectV2Class", "\345\270\270\347\224\250\350\250\255\345\256\232", nullptr));
        SaveLocationLabel->setText(QApplication::translate("DentistProjectV2Class", "\345\204\262\345\255\230\350\263\207\346\226\231\347\232\204\350\267\257\345\276\221\357\274\232", nullptr));
        SaveLocationBtn->setText(QApplication::translate("DentistProjectV2Class", "\351\201\270\346\223\207\350\267\257\345\276\221", nullptr));
        SaveWithTime_CheckBox->setText(QApplication::translate("DentistProjectV2Class", "\344\273\245\346\231\202\351\226\223\345\204\262\345\255\230", nullptr));
        AutoScanRawDataWhileScan_CheckBox->setText(QApplication::translate("DentistProjectV2Class", "\346\216\203\346\217\217\346\231\202\350\207\252\345\213\225\345\204\262\345\255\230 Raw Data", nullptr));
        ScanButton->setText(QApplication::translate("DentistProjectV2Class", "\346\216\203    \346\217\217    \346\250\241    \345\274\217", nullptr));
        AutoScanImageWhileScan_CheckBox->setText(QApplication::translate("DentistProjectV2Class", "\346\216\203\346\217\217\346\231\202\350\207\252\345\213\225\345\204\262\345\255\230\345\275\261\345\203\217\347\265\220\346\236\234", nullptr));
        OCTTestingBox->setTitle(QApplication::translate("DentistProjectV2Class", "OCT \346\270\254\350\251\246\347\233\270\351\227\234 (\351\200\262\351\232\216)", nullptr));
        RawDataToImage->setText(QApplication::translate("DentistProjectV2Class", "\350\274\270\345\207\272\345\234\226 (Raw Data)", nullptr));
        EasyBorderDetect->setText(QApplication::translate("DentistProjectV2Class", "\347\260\241\346\230\223\351\202\212\347\225\214\346\270\254\350\251\246", nullptr));
        RawDataCheck->setText(QApplication::translate("DentistProjectV2Class", "\350\274\270\345\207\272\350\263\207\346\226\231 (Raw Data)", nullptr));
        ShakeTestButton->setText(QApplication::translate("DentistProjectV2Class", "\346\231\203\345\213\225\345\201\265\346\270\254", nullptr));
        SegNetTestButton->setText(QApplication::translate("DentistProjectV2Class", "SegNet \351\240\220\346\270\254", nullptr));
        BeepSoundTestButton->setText(QApplication::translate("DentistProjectV2Class", "Beep Sound \346\270\254\350\251\246", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(Tab_OCT), QApplication::translate("DentistProjectV2Class", "OCT \350\243\235\347\275\256\350\250\255\345\256\232", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistProjectV2Class: public Ui_DentistProjectV2Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTPROJECTV2_H
