#version 430 core
// �ǤJ
layout(location = 0) in vec3 VertexPos;
layout(location = 1) in vec2 UV;

// �ǥX
out vec2 OutUV;

void main(void)
{
	gl_Position = vec4(VertexPos, 1.0);
	OutUV = UV;
}