#pragma once

#include <stack>

#include "../shapes/face.h"
#include "voxel-coordinates.h"

namespace Voxel
{
	template<size_t X, size_t Y, size_t Z>
	class VoxelSpace
	{
	public:
		static size_t DimX = X;
		static size_t DimY = Y;
		static size_t DimZ = Z;

		VoxelSpace()
		{
			for (int i = 0; i < X; i++)
				for (int j = 0; j < Y; j++)
					for (int k = 0; k < Z; k++)
						values[i][j][k] = std::numeric_limits<float>::max();
		}

		bool inside(VoxelCoordinates coordinates)
		{
			return
				coordinates.x >= 0 && coordinates.y >= 0 && coordinates.z >= 0 &&
				coordinates.x < X && coordinates.y < Y && coordinates.z < Z;
		}

		float& operator[](const VoxelCoordinates& coordinates)
		{
			return values[coordinates.x][coordinates.y][coordinates.z];
		}

	private:
		float values[X][Y][Z];
	};
}