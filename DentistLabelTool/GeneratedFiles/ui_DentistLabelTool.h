/********************************************************************************
** Form generated from reading UI file 'DentistLabelTool.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DENTISTLABELTOOL_H
#define UI_DENTISTLABELTOOL_H

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

class Ui_DentistLabelToolClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *DentistLabelToolClass)
    {
        if (DentistLabelToolClass->objectName().isEmpty())
            DentistLabelToolClass->setObjectName(QStringLiteral("DentistLabelToolClass"));
        DentistLabelToolClass->resize(600, 400);
        menuBar = new QMenuBar(DentistLabelToolClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        DentistLabelToolClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(DentistLabelToolClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        DentistLabelToolClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(DentistLabelToolClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        DentistLabelToolClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(DentistLabelToolClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        DentistLabelToolClass->setStatusBar(statusBar);

        retranslateUi(DentistLabelToolClass);

        QMetaObject::connectSlotsByName(DentistLabelToolClass);
    } // setupUi

    void retranslateUi(QMainWindow *DentistLabelToolClass)
    {
        DentistLabelToolClass->setWindowTitle(QApplication::translate("DentistLabelToolClass", "DentistLabelTool", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistLabelToolClass: public Ui_DentistLabelToolClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTLABELTOOL_H
