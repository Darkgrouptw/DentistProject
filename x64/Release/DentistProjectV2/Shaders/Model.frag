#version 430 core
// 輸入
in vec3 FragBaryCentricCoord;

// Const
const vec4 TriangleColor	= vec4(0.968f,	0.863f,	0.445f, 1);
const vec4 BorderColor		= vec4(0,		0,		0,		1);

// Helper Function
float edgeFactor(){
    vec3 d = fwidth(FragBaryCentricCoord);
    vec3 a3 = smoothstep(vec3(0.0), d*0.5, FragBaryCentricCoord);
    return min(min(a3.x, a3.y), a3.z);
}

void main()
{
	gl_FragColor = mix(BorderColor, TriangleColor, edgeFactor());
}