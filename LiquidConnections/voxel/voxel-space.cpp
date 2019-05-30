#include "voxel-space.h"

template<size_t X, size_t Y, size_t Z>
Voxel::VoxelSpace<X, Y, Z>::VoxelSpace()
{
	for (int i = 0; i < X; i++)
		for (int j = 0; j < Y; j++)
			for (int k = 0; k < Z; k++)
				values[i][j][k] = 0;
}

template<size_t X, size_t Y, size_t Z>
Voxel::VoxelSpace<X, Y, Z>::add(const Shapes::Face& face)
{
	
}