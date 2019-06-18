#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform samplerBuffer weights;
uniform mat4 transformation;

out vec3 fColor;

void main()
{
	float weight = texelFetch(weights, gl_InstanceID).r;
	vec3 coordinates = vec3(gl_InstanceID % 40 - 20, gl_InstanceID / 40 % 40 - 20, gl_InstanceID / 40 / 40 % 40 - 20);

    gl_Position = transformation * vec4((aPos * 0.3 * weight + coordinates) * 0.03, 1.0);
    fColor = aColor;
} 