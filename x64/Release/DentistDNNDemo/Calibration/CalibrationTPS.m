% 校正的矩陣
function UnDistort_Predict = CalibrationTPS(TPS, UnDistort_Calibration)
    % 算 TPS
	TPS_U_X = tpaps(UnDistort_Calibration(:, 1:2)', UnDistort_Calibration(:, 3)');
    TPS_U_Y = tpaps(UnDistort_Calibration(:, 1:2)', UnDistort_Calibration(:, 4)');

    % 預測
    UnDistort_Predict_X = fnval(TPS_U_X, TPS(:, 1:2)')';
    UnDistort_Predict_Y = fnval(TPS_U_Y, TPS(:, 1:2)')';
    [w, ~]= size(UnDistort_Predict_X);
    UnDistort_Predict = zeros([w, 2]);
    UnDistort_Predict(:, 1) = UnDistort_Predict_X;
    UnDistort_Predict(:, 2) = UnDistort_Predict_Y;
end