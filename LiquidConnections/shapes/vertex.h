#pragma once

namespace Shapes
{
	struct Vertex
	{
		float x;
		float y;
		float z;

		Vertex()
		{ }

		Vertex(float x, float y, float z) :
			x(x),
			y(y),
			z(z)
		{ }

		Vertex operator*(float by) const
		{
			return Vertex(x * by, y * by, z * by);
		}
	};
}