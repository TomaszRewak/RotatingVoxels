#pragma once

#include <array>

#include "ray.h"

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

		float minX() const;
		float minY() const;
		float minZ() const;

		float maxX() const;
		float maxY() const;
		float maxZ() const;

		bool intersect(const Ray& ray, Vertex& intersection) const;
	};
}