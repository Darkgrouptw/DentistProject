#pragma once
#include <iostream>
#include <QtWidgets/QMainWindow>
#include <QFileDialog>

#include "ui_DentistTrainningDataMaker.h"

using namespace std;

class DentistTrainningDataMaker : public QMainWindow
{
	Q_OBJECT

public:
	DentistTrainningDataMaker(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistTrainningDataMakerClass ui;


private slots:
	void LoadSTL();
};
