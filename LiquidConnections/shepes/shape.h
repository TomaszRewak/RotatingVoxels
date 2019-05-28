#pragma once

#include <vector>

#include "face.h"

namespace LiquidConnections
{
	namespace Shapes
	{
		struct Shape
		{
			std::size_t facesCount;
			Face* faces;
		};
	}
}