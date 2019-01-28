#version 430 core
// 輸入
layout(location = 0) in vec3 VertexPos;
layout(location = 1) in vec3 BaryCentricCoord;

// 輸出
out vec3 FragBaryCentricCoord;

// Uniform
uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ModelMatrix;

void main()
{
	// 傳進 Frag
	FragBaryCentricCoord = BaryCentricCoord;

	// Position
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(VertexPos, 1);
}