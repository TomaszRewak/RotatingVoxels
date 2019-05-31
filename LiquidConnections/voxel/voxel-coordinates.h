#pragma once

namespace Voxel
{
	struct VoxelCoordinates
	{
		int x;
		int y;
		int z;

		VoxelCoordinates() :
			x(0),
			y(0),
			z(0)
		{ }

		VoxelCoordinates(int x, int y, int z) :
			x(x),
			y(y),
			z(z)
		{ }

		VoxelCoordinates move(int dx, int dy, int dz)
		{
			return VoxelCoordinates(x + dx, y + dy, z + dz);
		}
	};
}