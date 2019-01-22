#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DentistProjectV2.h"

class DentistProjectV2 : public QMainWindow
{
	Q_OBJECT

public:
	DentistProjectV2(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistProjectV2Class ui;
};
