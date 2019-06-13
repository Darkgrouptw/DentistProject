#version 430 core
// 傳入
in vec2 OutUV;

// 貼圖等 Uniform 參數
uniform sampler2D texture;
uniform sampler2D probTexture;

// 輸出
out vec4 FragColor;

void main()
{
	// FragColor = texture2D(texture, OutUV) + texture2D(probTexture, OutUV);
	FragColor = texture2D(probTexture, OutUV);
	//FragColor = vec4(1,1,1,1);
}