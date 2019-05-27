#pragma once

#include <array>

#include "vertex.h"
#include "vector.h"

namespace LiquidConnections
{
	namespace Shapes
	{
		struct Face
		{
			std::array<Vertex, 3> vertices;
			Vector normal;

			Face(Vertex a, Vertex b, Vertex c, Vector normal) :
				vertices{ a, b, c },
				normal(normal)
			{ }
		};
	}
}