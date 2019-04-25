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
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QWidget>
#include "display_topview.h"

QT_BEGIN_NAMESPACE

class Ui_DentistLabelToolClass
{
public:
    QAction *actionOpen;
    QWidget *centralWidget;
    Display_TopView *Widget1;
    QOpenGLWidget *Widget2;
    QScrollBar *SliderPage;
    QMenuBar *menuBar;
    QMenu *menu;

    void setupUi(QMainWindow *DentistLabelToolClass)
    {
        if (DentistLabelToolClass->objectName().isEmpty())
            DentistLabelToolClass->setObjectName(QStringLiteral("DentistLabelToolClass"));
        DentistLabelToolClass->resize(1200, 600);
        actionOpen = new QAction(DentistLabelToolClass);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        centralWidget = new QWidget(DentistLabelToolClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        Widget1 = new Display_TopView(centralWidget);
        Widget1->setObjectName(QStringLiteral("Widget1"));
        Widget1->setGeometry(QRect(0, 0, 500, 500));
        Widget2 = new QOpenGLWidget(centralWidget);
        Widget2->setObjectName(QStringLiteral("Widget2"));
        Widget2->setGeometry(QRect(500, 0, 500, 500));
        SliderPage = new QScrollBar(centralWidget);
        SliderPage->setObjectName(QStringLiteral("SliderPage"));
        SliderPage->setGeometry(QRect(0, 500, 500, 16));
        SliderPage->setOrientation(Qt::Horizontal);
        DentistLabelToolClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(DentistLabelToolClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1200, 21));
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
