#version 430 core
// �ǤJ
in vec2 OutUV;

// �K�ϵ� Uniform �Ѽ�
uniform sampler2D texture;
uniform sampler2D probTexture;

// ��X
out vec4 FragColor;

void main()
{
	// FragColor = texture2D(texture, OutUV) + texture2D(probTexture, OutUV);
	FragColor = texture2D(probTexture, OutUV);
	//FragColor = vec4(1,1,1,1);
}