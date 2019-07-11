% 清空結果
clc;clear;close all;

% 讀檔案
Calibration = csvread("Calibration.csv");
Valid_Calibration = csvread("Valid_Calibration.csv");
UnDistort_Calibration = csvread("UnDistort_Calibration.csv");
UnDistort_Valid_Calibration = csvread("UnDistort_Valid_Calibration.csv");

% TPS
TPS_X = tpaps(Calibration(:, 1:2)', Calibration(:, 3)');
TPS_Y = tpaps(Calibration(:, 1:2)', Calibration(:, 4)');

% Prdict
Predict_X = fnval(TPS_X, Valid_Calibration(:, 1:2)')';
Predict_Y = fnval(TPS_Y, Valid_Calibration(:, 1:2)')';
[w, ~]= size(Predict_X);
Predict = zeros([w, 2]);
Predict(:, 1) = Predict_X;
Predict(:, 2) = Predict_Y;

% Camera Calibration (只算 TransformMatrix 矩陣)
[FullW, ~] = size(Calibration);
PixelPoint = zeros([FullW, 4]);
WorldPoint = zeros([FullW, 4]);
PixelPoint(:, 4) = 1;
PixelPoint(:, 1:2) = Calibration(:, 1:2);
WorldPoint(:, 4) = 1;
WorldPoint(:, 1:2) = Calibration(:, 3:4);
TransformMatrix = pinv(PixelPoint' * PixelPoint) * (PixelPoint' * WorldPoint);

% Predict
CalibrationPredict = zeros([w, 4]);
CalibrationPredict(:, 1:2) = Valid_Calibration(:, 1:2);
CalibrationPredict(:, 4) = 1;
CalibrationPredict = CalibrationPredict * TransformMatrix;

% UnDistor
% TPS
TPS_U_X = tpaps(UnDistort_Calibration(:, 1:2)', UnDistort_Calibration(:, 3)');
TPS_U_Y = tpaps(UnDistort_Calibration(:, 1:2)', UnDistort_Calibration(:, 4)');

% Prdict
UnDistort_Predict_X = fnval(TPS_U_X, UnDistort_Valid_Calibration(:, 1:2)')';
UnDistort_Predict_Y = fnval(TPS_U_Y, UnDistort_Valid_Calibration(:, 1:2)')';
UnDistort_Predict = zeros([w, 2]);
UnDistort_Predict(:, 1) = UnDistort_Predict_X;
UnDistort_Predict(:, 2) = UnDistort_Predict_Y;

% 畫出點來
figure();
hold on;
axis([1, 10, 2, 10]);
scatter(Valid_Calibration(:, 3), Valid_Calibration(:, 4), 'ko');
scatter(Predict(:, 1), Predict(:, 2), 'rx');
scatter(CalibrationPredict(:, 1), CalibrationPredict(:, 2), 'g*')
scatter(Valid_Calibration(:, 3), Valid_Calibration(:, 4), 'b+');
hold off;

% 算 Error 
Error_Predict = ((Valid_Calibration(:, 3) - Predict(:, 1)).^2 + ((Valid_Calibration(:, 4) - Predict(:, 2)).^2)).^0.5;
Error_Calibration = ((Valid_Calibration(:, 3) - CalibrationPredict(:, 1)).^2 + ((Valid_Calibration(:, 4) - CalibrationPredict(:, 2)).^2)).^0.5;
Error_UnDistort = ((Valid_Calibration(:, 3) - UnDistort_Predict(:, 1)).^2 + ((Valid_Calibration(:, 4) - UnDistort_Predict(:, 2)).^2)).^0.5;

figure();
hold on;
histogram(Error_Predict, 20, "FaceColor",'r');
histogram(Error_Calibration, 20, "FaceColor", 'g');
histogram(Error_UnDistort, 20, "FaceColor", 'b');
hold off;

