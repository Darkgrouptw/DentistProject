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
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DentistLabelToolClass
{
public:
    QAction *actionOpen;
    QWidget *centralWidget;
    QOpenGLWidget *openGLWidget;
    QMenuBar *menuBar;
    QMenu *menu;

    void setupUi(QMainWindow *DentistLabelToolClass)
    {
        if (DentistLabelToolClass->objectName().isEmpty())
            DentistLabelToolClass->setObjectName(QStringLiteral("DentistLabelToolClass"));
        DentistLabelToolClass->resize(1500, 900);
        actionOpen = new QAction(DentistLabelToolClass);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        centralWidget = new QWidget(DentistLabelToolClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        openGLWidget = new QOpenGLWidget(centralWidget);
        openGLWidget->setObjectName(QStringLiteral("openGLWidget"));
        openGLWidget->setGeometry(QRect(0, 0, 900, 900));
        DentistLabelToolClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(DentistLabelToolClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1500, 21));
        menu = new QMenu(menuBar);
        menu->setObjectName(QStringLiteral("menu"));
        DentistLabelToolClass->setMenuBar(menuBar);

        menuBar->addAction(menu->menuAction());
        menu->addAction(actionOpen);

        retranslateUi(DentistLabelToolClass);

        QMetaObject::connectSlotsByName(DentistLabelToolClass);
    } // setupUi

    void retranslateUi(QMainWindow *DentistLabelToolClass)
    {
        DentistLabelToolClass->setWindowTitle(QApplication::translate("DentistLabelToolClass", "DentistLabelTool", nullptr));
        actionOpen->setText(QApplication::translate("DentistLabelToolClass", "Open", nullptr));
        menu->setTitle(QApplication::translate("DentistLabelToolClass", "\346\252\224\346\241\210", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DentistLabelToolClass: public Ui_DentistLabelToolClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTLABELTOOL_H
