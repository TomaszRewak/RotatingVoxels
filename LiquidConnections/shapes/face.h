#pragma once

#include <array>

#include "vertex.h"
#include "vector.h"

namespace Shapes
{
	struct Face
	{
		Vertex vertices[3];
		Vector normal;

		Face()
		{ }

		Face(Vertex a, Vertex b, Vertex c, Vector normal) :
			vertices{ a, b, c },
			normal(normal)
		{ }

		bool intersect(const Vector& vector, Vertex& intersection) const;
	};
}