#version 430 core
// 傳入
in vec2 OutUV;

// 貼圖等 Uniform 參數
uniform sampler2D texture;
uniform sampler2D probTexture;
uniform sampler2D colorMapTexture;
uniform float tempuv;

// 輸出
out vec4 FragColor;

const float Threshold = 0.2f;

// 抓取灰階值
float GrayScaleValue(vec4 color)
{
	return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

void main()
{
	// 抓顏色
	vec4 Ori = texture2D(texture, OutUV);
	vec4 Prob = texture2D(probTexture, OutUV);
	vec4 Color = texture2D(colorMapTexture, OutUV);
	
	// 總和
	FragColor = Ori;

	if (GrayScaleValue(Color) > Threshold)
		FragColor += Color;
		
	if (GrayScaleValue(Prob) > Threshold){
		if(OutUV.x>tempuv)
			return;
		FragColor += Prob;
	}
}