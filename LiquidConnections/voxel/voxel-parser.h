#pragma once

#include "voxel-space.h"
#include "../shapes/shape.h"

namespace Voxel
{
	namespace VoxelParser
	{
		template<size_t X, size_t Y, size_t Z>
		Shapes::Shape createShape(const VoxelSpace<X, Y, Z>& voxelSpace)
		{
			Shapes::Shape shape;

			for (int x = 0; x < X; x++)
				for (int y = 0; y < Y; y++)
					for (int z = 0; z < Z; z++)
						if (voxelSpace[VoxelCoordinates(x, y, z)] <= 0)
							shape.faces.push_back(Shapes::Face(
								Shapes::Vertex(x, y, z),
								Shapes::Vertex(x + 1, y, z),
								Shapes::Vertex(x, y + 1, z),
								Shapes::Vector(0, 0, -1)
							));

			return shape;
		}
	}
}