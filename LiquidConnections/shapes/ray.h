#pragma once

#include "vector.h"

namespace Shapes
{
	struct Ray
	{
		Vertex origin;
		Vector vector;

		Ray(Vertex origin, Vector vector) :
			origin(origin),
			vector(vector)
		{ }

		Vertex intersect(float distance) const;
	};
}