#pragma once

#include <vector>

#include "face.h"

namespace LiquidConnections
{
	namespace Shapes
	{
		struct Shape
		{
			std::vector<Face> faces;
		};
	}
}