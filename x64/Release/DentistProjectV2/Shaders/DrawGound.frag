#version 430 core
// 輸入
in vec2 UV;

// Const
const vec4 TriangleColor	= vec4(0.7,	0.7f,	0.7f, 	1);
const vec4 BorderColor		= vec4(1,	1,		1,		1);
const int SplitNum = 50;
const int GapNum = 2;

// Helper Function
float smoothFunction(int value)
{
	int modValue = value % SplitNum;
	float smoothValue = 0;
	if (modValue > SplitNum / 2)
		smoothValue = smoothstep(SplitNum - GapNum, SplitNum, modValue);
	else if (modValue < SplitNum / 2)
		smoothValue = smoothstep(SplitNum - GapNum, SplitNum, SplitNum - modValue);
	return smoothValue;
}
float edgeFactor()
{
	// 放大
	vec2 ScaleUV = UV * 1000;
	
	int x = int(ScaleUV.x);
	int y = int(ScaleUV.y);
	
	float xCord = smoothFunction(x);
	float yCord = smoothFunction(y);
	return max(xCord, yCord);
}

void main()
{
	gl_FragColor = mix(TriangleColor, BorderColor, edgeFactor());
}