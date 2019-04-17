#pragma once
#include "TensorflowNet.h"

#include <QtWidgets/QMainWindow>
#include "ui_DentistLabelTool.h"

class DentistLabelTool : public QMainWindow
{
	Q_OBJECT

public:
	DentistLabelTool(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistLabelToolClass ui;
};
