#pragma once

#include <algorithm>
#include <limits>

#include "../shapes/shape.h"

namespace Voxel
{
	namespace VoxelShapePreprocesor
	{
		template<size_t X, size_t Y, size_t Z>
		Shapes::Shape normalizeShape(Shapes::Shape shape)
		{
			std::random_shuffle(shape.faces.begin(), shape.faces.end());

			float
				minX = std::numeric_limits<float>::max(),
				minY = std::numeric_limits<float>::max(),
				minZ = std::numeric_limits<float>::max(),
				maxX = std::numeric_limits<float>::min(),
				maxY = std::numeric_limits<float>::min(),
				maxZ = std::numeric_limits<float>::min();

			for (const auto& face : shape.faces)
			{
				minX = std::min(minX, face.minX());
				minY = std::min(minY, face.minY());
				minZ = std::min(minZ, face.minZ());

				maxX = std::max(maxX, face.maxX());
				maxY = std::max(maxY, face.maxY());
				maxZ = std::max(maxZ, face.maxZ());
			}

			float
				scaleX = X / (maxX - minX),
				scaleY = Y / (maxY - minY),
				scaleZ = Z / (maxZ - minZ);

			auto offset = Shapes::Vector(-minX, -minY, -minZ);
			float scale = std::min(std::min(scaleX, scaleY), scaleZ);

			for (auto& face : shape.faces)
			{
				face.vertices[0] = (face.vertices[0] + offset) * scale;
				face.vertices[1] = (face.vertices[1] + offset) * scale;
				face.vertices[2] = (face.vertices[2] + offset) * scale;
			}

			return shape;
		}
	};
}