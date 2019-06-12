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
    QScrollBar *horizontalScrollBar;
    QGroupBox *groupBox;
    QPushButton *TestRenderingBtn;

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
        horizontalScrollBar = new QScrollBar(centralWidget);
        horizontalScrollBar->setObjectName(QStringLiteral("horizontalScrollBar"));
        horizontalScrollBar->setGeometry(QRect(0, 500, 501, 16));
        horizontalScrollBar->setOrientation(Qt::Horizontal);
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(1020, 280, 171, 221));
        TestRenderingBtn = new QPushButton(groupBox);
        TestRenderingBtn->setObjectName(QStringLiteral("TestRenderingBtn"));
        TestRenderingBtn->setGeometry(QRect(10, 20, 151, 31));
        DentistDNNDemoClass->setCentralWidget(centralWidget);

        retranslateUi(DentistDNNDemoClass);

        QMetaObject::connectSlotsByName(DentistDNNDemoClass);
    } // setupUi

    void retranslateUi(QMainWindow *DentistDNNDemoClass)
    {
        DentistDNNDemoClass->setWindowTitle(QApplication::translate("DentistDNNDemoClass", "DentistDNNDemo", nullptr));
        groupBox->setTitle(QApplication::translate("DentistDNNDemoClass", "Test Function", nullptr));
        TestRenderingBtn->setText(QApplication::translate("DentistDNNDemoClass", "\346\270\254\350\251\246 Rendering \347\232\204 Function", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistDNNDemoClass: public Ui_DentistDNNDemoClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTDNNDEMO_H
