#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

layout (std430, binding=3) buffer shader_data
{
	float weights[];
};

out vec3 fColor;

void main()
{
	float weight = weights[0];
	vec3 coordinates = vec3(gl_InstanceID % 40 - 20, gl_InstanceID / 40 % 40 - 20, gl_InstanceID / 40 / 40 % 40 - 20);

    gl_Position = vec4((aPos * 0.3 * weight + coordinates) * 0.03, 1.0);
    fColor = aColor;
} 