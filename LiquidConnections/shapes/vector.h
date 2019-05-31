#pragma once

#include "vertex.h"

namespace Shapes
{
	struct Vector
	{
		float x;
		float y;
		float z;

		Vector()
		{ }

		Vector(float x, float y, float z) :
			x(x),
			y(y),
			z(z)
		{ }

		Vector(const Vertex& begin, const Vertex& end) :
			x(end.x - begin.x),
			y(end.y - begin.y),
			z(end.z - begin.z)
		{ }

		float dotProduct(const Vector& second) const;
		Vector crossProduct(const Vector& second) const;

		Vector operator*(float by) const;
	};

	Vertex operator+(const Vertex& vertex, const Vector& vector);
}