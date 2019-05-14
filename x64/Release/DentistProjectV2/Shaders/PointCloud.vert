#version 430 core
// 輸入
layout(location = 0) in vec3 VertexPos;

// Uniform Matrix
uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrix;

// Uniform Point Size
uniform float pointSize;

void main()
{
	// 點的大小
	gl_PointSize = pointSize;
	
	// Position
	gl_Position = ProjectionMatrix * ViewMatrix * vec4(VertexPos, 1);
}