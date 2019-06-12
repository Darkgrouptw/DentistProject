#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DentistDNNDemo.h"

class DentistDNNDemo : public QMainWindow
{
	Q_OBJECT

public:
	DentistDNNDemo(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistDNNDemoClass ui;
};
