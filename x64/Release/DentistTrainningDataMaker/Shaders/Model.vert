#version 430 core
// ��J
layout(location = 0) in vec3 VertexPos;
layout(location = 1) in vec3 BaryCentricCoord;

// ��X
out vec3 FragBaryCentricCoord;

// Uniform
uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ModelMatrix;

void main()
{
	// �Ƕi Frag
	FragBaryCentricCoord = BaryCentricCoord;

	// Position
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(VertexPos, 1);
}