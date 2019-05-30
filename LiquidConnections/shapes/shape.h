#pragma once

#include <vector>

#include "face.h"

namespace Shapes
{
	struct Shape
	{
		std::size_t facesCount;
		std::vector<Face> faces;
	};
}