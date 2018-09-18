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
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DentistProjectClass
{
public:
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *DentistProjectClass)
    {
        if (DentistProjectClass->objectName().isEmpty())
            DentistProjectClass->setObjectName(QStringLiteral("DentistProjectClass"));
        DentistProjectClass->resize(1600, 900);
        centralWidget = new QWidget(DentistProjectClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        DentistProjectClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(DentistProjectClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        DentistProjectClass->setStatusBar(statusBar);

        retranslateUi(DentistProjectClass);

        QMetaObject::connectSlotsByName(DentistProjectClass);
    } // setupUi

    void retranslateUi(QMainWindow *DentistProjectClass)
    {
        DentistProjectClass->setWindowTitle(QApplication::translate("DentistProjectClass", "DentistProject", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistProjectClass: public Ui_DentistProjectClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTPROJECT_H
