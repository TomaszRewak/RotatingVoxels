#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform samplerBuffer weights;
uniform mat4 transformation;
uniform vec3 size;

out vec3 fColor;
out vec3 fPos;
out float fWeight;

mat4 lookAt(inout vec3 target)
{
	vec3 forward = normalize(target);
	vec3 left = normalize(cross(vec3(0, 1, 0), forward));
	vec3 up = cross(forward, left);

	mat4 matrix = mat4(1.0f);

	matrix[0][0] = left.x;
	matrix[0][1] = left.y;
	matrix[0][2] = left.z;
	matrix[1][0] = up.x;
	matrix[1][1] = up.y;
	matrix[1][2] = up.z;
	matrix[2][0] = forward.x;
	matrix[2][1] = forward.y;
	matrix[2][2] = forward.z;

	return matrix;
}

void main()
{
	vec4 texel = texelFetch(weights, gl_InstanceID);

	float weight = texel.r;
	vec3 normal = vec3(texel.g, texel.b, texel.a);

	int x = gl_InstanceID                             % int(size.x);
	int y = gl_InstanceID / int(size.x)               % int(size.y);
	int z = gl_InstanceID / int(size.x) / int(size.y) % int(size.z);

	vec4 coordinates = vec4(
		x - size.x / 2, 
		y - size.y / 2, 
		z - size.z / 2, 
		size.x);

	if (x % 5 == 0 && y % 5 == 0)
	{
		fWeight = weight = max(weight, 0.2);
		fColor = aColor * vec3(1.0, 0.5, 0.5);
	} else
	{	
		fWeight = weight;
		fColor = aColor;
	}


	mat4 direction = lookAt(normal);
	vec4 cornerPos = vec4(aPos * weight * 0.3, 20);

    gl_Position = transformation * (coordinates + direction * cornerPos);
	
	fPos = aPos;
} 