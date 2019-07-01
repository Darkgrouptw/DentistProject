#version 430 core
// �ǤJ
in vec2 OutUV;

// �K�ϵ� Uniform �Ѽ�
uniform sampler2D texture;
uniform sampler2D probTexture;
uniform sampler2D colorMapTexture;

// ��X
out vec4 FragColor;

const float Threshold = 0.2f;

// ����Ƕ���
float GrayScaleValue(vec4 color)
{
	return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

void main()
{
	// ���C��
	vec4 Ori = texture2D(texture, OutUV);
	vec4 Prob = texture2D(probTexture, OutUV);
	vec4 Color = texture2D(colorMapTexture, OutUV);
	
	// �`�M
	FragColor = Ori;
	if (GrayScaleValue(Prob) > Threshold)
		FragColor += Prob;
	//if (GrayScaleValue(Color) > Threshold)
	//	FragColor.rgb = FragColor.rgb * 0.8f + Color.rgb * 0.2f;
}