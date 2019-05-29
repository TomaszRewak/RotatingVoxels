#pragma once

namespace LiquidConnections
{
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
		};
	}
}