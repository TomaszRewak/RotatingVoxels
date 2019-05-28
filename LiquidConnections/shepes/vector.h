#pragma once

namespace LiquidConnections
{
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
		};
	}
}