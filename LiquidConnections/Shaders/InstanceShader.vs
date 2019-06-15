#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 fColor;

void main()
{
    gl_Position = vec4(aPos.x * 0.5 + gl_InstanceID * 0.3, aPos.y * 0.5 + gl_InstanceID * 0.3, aPos.z * 0.5, 1.0);
    fColor = aColor;
} 