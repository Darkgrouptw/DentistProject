#version 430 core
// 輸入
layout(location = 0) in vec3 VertexPos;
layout(location = 1) in vec2 VertexUV;

// 輸出
out vec2 UV;

// Uniform
uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ModelMatrix;

void main()
{
	UV = VertexUV;

	// Position
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(VertexPos, 1);
}