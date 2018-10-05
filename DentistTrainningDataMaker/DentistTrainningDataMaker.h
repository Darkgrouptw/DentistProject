#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DentistTrainningDataMaker.h"

class DentistTrainningDataMaker : public QMainWindow
{
	Q_OBJECT

public:
	DentistTrainningDataMaker(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistTrainningDataMakerClass ui;
};
