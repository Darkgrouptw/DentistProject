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
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QStatusBar>
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
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

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
        RenderOptionsBox->setGeometry(QRect(1490, 10, 91, 211));
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
        DentistTrainningDataMakerClass->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(DentistTrainningDataMakerClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        DentistTrainningDataMakerClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(DentistTrainningDataMakerClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        DentistTrainningDataMakerClass->setStatusBar(statusBar);

        mainToolBar->addAction(actionLoadSTL);

        retranslateUi(DentistTrainningDataMakerClass);

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
    } // retranslateUi

};

namespace Ui {
    class DentistTrainningDataMakerClass: public Ui_DentistTrainningDataMakerClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENTISTTRAINNINGDATAMAKER_H
