#version 430 core
// 傳入
layout(location = 0) in vec3 VertexPos;
layout(location = 1) in vec2 UV;

// 傳出
out vec2 OutUV;

void main(void)
{
	gl_Position = vec4(VertexPos, 1.0);
	OutUV = UV;
}