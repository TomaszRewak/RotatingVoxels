#pragma once

#include "../shapes/shape.h"

namespace Voxel
{
	namespace VoxelShapePreprocesor
	{
		Shapes::Shape normalizeShape(Shapes::Shape shape)
		{
			std::shuffle(shape.faces.begin(), shape.faces.end())
		}
	};
}