/********************************************************************************
** Form generated from reading UI file 'DentistDNNDemo.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
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
    QLabel *ColorMapLabel;
    QLabel *ColorMapMaxValue;
    QLabel *ColorMapMinValue;
    QLabel *ColorMapCurrentValue;

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
        ColorMapLabel = new QLabel(centralWidget);
        ColorMapLabel->setObjectName(QStringLiteral("ColorMapLabel"));
        ColorMapLabel->setGeometry(QRect(510, 40, 16, 401));
        ColorMapMaxValue = new QLabel(centralWidget);
        ColorMapMaxValue->setObjectName(QStringLiteral("ColorMapMaxValue"));
        ColorMapMaxValue->setGeometry(QRect(530, 40, 47, 12));
        ColorMapMinValue = new QLabel(centralWidget);
        ColorMapMinValue->setObjectName(QStringLiteral("ColorMapMinValue"));
        ColorMapMinValue->setGeometry(QRect(530, 430, 47, 12));
        ColorMapCurrentValue = new QLabel(centralWidget);
        ColorMapCurrentValue->setObjectName(QStringLiteral("ColorMapCurrentValue"));
        ColorMapCurrentValue->setGeometry(QRect(530, 260, 47, 12));
        DentistDNNDemoClass->setCentralWidget(centralWidget);

        retranslateUi(DentistDNNDemoClass);

        QMetaObject::connectSlotsByName(DentistDNNDemoClass);
    } // setupUi

    void retranslateUi(QMainWindow *DentistDNNDemoClass)
    {
        DentistDNNDemoClass->setWindowTitle(QApplication::translate("DentistDNNDemoClass", "DentistDNNDemo", nullptr));
        groupBox->setTitle(QApplication::translate("DentistDNNDemoClass", "Test Function", nullptr));
        TestRenderingBtn->setText(QApplication::translate("DentistDNNDemoClass", "\346\270\254\350\251\246 Rendering \347\232\204 Function", nullptr));
        ColorMapLabel->setText(QString());
        ColorMapMaxValue->setText(QString());
        ColorMapMinValue->setText(QString());
        ColorMapCurrentValue->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class DentistDNNDemoClass: public Ui_DentistDNNDemoClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTDNNDEMO_H
