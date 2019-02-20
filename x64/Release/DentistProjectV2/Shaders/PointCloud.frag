#version 430 core
// Point Color
const vec4 CurrentPC_Color = vec4(0, 0, 1, 1);
const vec4 NonCurrentPC_Color = new vec4(0, 0, 0, 1);

// Uniform Color 
uniform bool IsCurrentPC;

void main()
{
	if (IsCurrentPC)
		gl_FragColor = CurrentPC_Color;
	else
		gl_FragColor = NonCurrentPC_Color;
}