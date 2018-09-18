#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DentistProject.h"

class DentistProject : public QMainWindow
{
	Q_OBJECT

public:
	DentistProject(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistProjectClass ui;
};
