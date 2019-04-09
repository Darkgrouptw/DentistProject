#version 430 core
// 輸入
layout(location = 0) in vec3 VertexPos;
layout(location = 1) in float PointType;

// 輸出
out vec3 FragBaryCentricCoord;
out vec3 VertexColor;

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


// 根據 Type 去拿顏色
vec3 TeethColor = vec3(1, 1, 1);
vec3 MeatColor = vec3(0.4, 0.2, 0.2);

void main()
{
	// 傳進 Frag
	FragBaryCentricCoord = BaryCentricCoord[gl_VertexID % 3];

	if(abs(PointType - 1) <= 0.01)
		VertexColor = TeethColor;
	else
		VertexColor = MeatColor;

	// Position
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(VertexPos, 1);
}