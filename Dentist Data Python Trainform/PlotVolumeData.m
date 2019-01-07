FileLocation = 'D:/Dentist/Data/ScanData/2018.10.18/20181016_Incisor_Label/';
% VolumeData = zeros(250, 250, 1024);

% ³]©w
StartIndex = 60;
EndIndex = 200;
Size = (EndIndex - StartIndex) * 250;

% Value
xValues = zeros(EndIndex - StartIndex, 1);
yValues = zeros(EndIndex - StartIndex, 1);
zValues = zeros(EndIndex - StartIndex, 1);

% Åª¹Ï
for x = StartIndex:EndIndex
    ImgLocation = strcat(FileLocation, int2str(x));
    ImgLocation = strcat(ImgLocation, '.png');
    TempImg = imread(ImgLocation);
    
%     for y = 1:250
%         for 
%     end
end
surf(xValues, yValues, zValues);
xlabel('x');
ylabel('y');
zlabel('Data');