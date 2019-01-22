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
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DentistProjectV2Class
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *DentistProjectV2Class)
    {
        if (DentistProjectV2Class->objectName().isEmpty())
            DentistProjectV2Class->setObjectName(QStringLiteral("DentistProjectV2Class"));
        DentistProjectV2Class->resize(600, 400);
        menuBar = new QMenuBar(DentistProjectV2Class);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        DentistProjectV2Class->setMenuBar(menuBar);
        mainToolBar = new QToolBar(DentistProjectV2Class);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        DentistProjectV2Class->addToolBar(mainToolBar);
        centralWidget = new QWidget(DentistProjectV2Class);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        DentistProjectV2Class->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(DentistProjectV2Class);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        DentistProjectV2Class->setStatusBar(statusBar);

        retranslateUi(DentistProjectV2Class);

        QMetaObject::connectSlotsByName(DentistProjectV2Class);
    } // setupUi

    void retranslateUi(QMainWindow *DentistProjectV2Class)
    {
        DentistProjectV2Class->setWindowTitle(QApplication::translate("DentistProjectV2Class", "DentistProjectV2", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistProjectV2Class: public Ui_DentistProjectV2Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTPROJECTV2_H
