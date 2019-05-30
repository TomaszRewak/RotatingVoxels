#pragma once

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

		float dotProduct(const Vector& second) const;
		Vector crossProduct(const Vector& second) const;
	};
}