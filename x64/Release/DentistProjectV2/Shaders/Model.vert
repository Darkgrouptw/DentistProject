#version 430 core
// 輸入
layout(location = 0) in vec3 VertexPos;

// 輸出
out vec3 FragBaryCentricCoord;

// Uniform
uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ModelMatrix;

// const
vec3 BaryCentricCoord[3] = {
	vec3(1, 0, 0),
	vec3(0, 1, 0),
	vec3(0, 0, 1)
};

void main()
{
	// 傳進 Frag
	FragBaryCentricCoord = BaryCentricCoord[gl_VertexID % 3];

	// Position
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(VertexPos, 1);
}