/********************************************************************************
** Form generated from reading UI file 'DentistDNNDemo.ui'
**
** Created by: Qt User Interface Compiler version 5.9.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DENTISTDNNDEMO_H
#define UI_DENTISTDNNDEMO_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QWidget>
#include "openglwidget.h"

QT_BEGIN_NAMESPACE

class Ui_DentistDNNDemoClass
{
public:
    QWidget *centralWidget;
    OpenGLWidget *DisplayPanel;
    QScrollBar *slidingBar;
    QGroupBox *groupBox;
    QPushButton *TestRenderingBtn;
    QPushButton *TestReadRawDataBtn;
    QPushButton *TestValidDataBtn;
    QLabel *ColorMapMaxValue;
    QLabel *ColorValue;
    QPushButton *ShowValueBtn;
    QLabel *ColorMapMinValue;
    QLabel *ColorMapLabel;

    void setupUi(QMainWindow *DentistDNNDemoClass)
    {
        if (DentistDNNDemoClass->objectName().isEmpty())
            DentistDNNDemoClass->setObjectName(QStringLiteral("DentistDNNDemoClass"));
        DentistDNNDemoClass->resize(1200, 550);
        centralWidget = new QWidget(DentistDNNDemoClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        DisplayPanel = new OpenGLWidget(centralWidget);
        DisplayPanel->setObjectName(QStringLiteral("DisplayPanel"));
        DisplayPanel->setGeometry(QRect(0, 0, 500, 500));
        slidingBar = new QScrollBar(centralWidget);
        slidingBar->setObjectName(QStringLiteral("slidingBar"));
        slidingBar->setGeometry(QRect(0, 500, 501, 16));
        slidingBar->setMinimum(60);
        slidingBar->setMaximum(200);
        slidingBar->setOrientation(Qt::Horizontal);
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(1020, 280, 171, 221));
        TestRenderingBtn = new QPushButton(groupBox);
        TestRenderingBtn->setObjectName(QStringLiteral("TestRenderingBtn"));
        TestRenderingBtn->setGeometry(QRect(10, 20, 151, 31));
        TestReadRawDataBtn = new QPushButton(groupBox);
        TestReadRawDataBtn->setObjectName(QStringLiteral("TestReadRawDataBtn"));
        TestReadRawDataBtn->setGeometry(QRect(10, 60, 151, 31));
        TestValidDataBtn = new QPushButton(groupBox);
        TestValidDataBtn->setObjectName(QStringLiteral("TestValidDataBtn"));
        TestValidDataBtn->setGeometry(QRect(10, 100, 151, 31));
        ColorMapMaxValue = new QLabel(centralWidget);
        ColorMapMaxValue->setObjectName(QStringLiteral("ColorMapMaxValue"));
        ColorMapMaxValue->setGeometry(QRect(530, 30, 47, 12));
        ColorValue = new QLabel(centralWidget);
        ColorValue->setObjectName(QStringLiteral("ColorValue"));
        ColorValue->setEnabled(true);
        ColorValue->setGeometry(QRect(570, 220, 91, 51));
        ColorValue->setStyleSheet(QStringLiteral("background-color: rgba(255, 255, 255,0.5);"));
        ShowValueBtn = new QPushButton(centralWidget);
        ShowValueBtn->setObjectName(QStringLiteral("ShowValueBtn"));
        ShowValueBtn->setGeometry(QRect(570, 40, 91, 41));
        ColorMapMinValue = new QLabel(centralWidget);
        ColorMapMinValue->setObjectName(QStringLiteral("ColorMapMinValue"));
        ColorMapMinValue->setGeometry(QRect(530, 480, 47, 12));
        ColorMapLabel = new QLabel(centralWidget);
        ColorMapLabel->setObjectName(QStringLiteral("ColorMapLabel"));
        ColorMapLabel->setGeometry(QRect(520, 50, 21, 421));
        DentistDNNDemoClass->setCentralWidget(centralWidget);

        retranslateUi(DentistDNNDemoClass);

        QMetaObject::connectSlotsByName(DentistDNNDemoClass);
    } // setupUi

    void retranslateUi(QMainWindow *DentistDNNDemoClass)
    {
        DentistDNNDemoClass->setWindowTitle(QApplication::translate("DentistDNNDemoClass", "DentistDNNDemo", Q_NULLPTR));
        groupBox->setTitle(QApplication::translate("DentistDNNDemoClass", "Test Function", Q_NULLPTR));
        TestRenderingBtn->setText(QApplication::translate("DentistDNNDemoClass", "\346\270\254\350\251\246 Rendering \347\232\204 Function", Q_NULLPTR));
        TestReadRawDataBtn->setText(QApplication::translate("DentistDNNDemoClass", "\350\256\200\345\217\226RawData \350\267\221\347\266\262\350\267\257", Q_NULLPTR));
        TestValidDataBtn->setText(QApplication::translate("DentistDNNDemoClass", "\346\270\254\350\251\246ValidData\347\224\250", Q_NULLPTR));
        ColorMapMaxValue->setText(QString());
        ColorValue->setText(QString());
        ShowValueBtn->setText(QApplication::translate("DentistDNNDemoClass", "ShowValue", Q_NULLPTR));
        ColorMapMinValue->setText(QString());
        ColorMapLabel->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class DentistDNNDemoClass: public Ui_DentistDNNDemoClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTDNNDEMO_H
