#pragma once

#include "../shapes/face.h"

namespace LiquidConnections
{
	namespace Voxel
	{
		template<size_t X, size_t Y, size_t Z>
		class VoxelSpace
		{
		public:
			static size_t x = X;
			static size_t y = Y;
			static size_t z = Z;

			VoxelSpace();

			void add(const Shapes::Face& face);
			void intersect(const Shapes::Vector& ray);

		private:
			float values[X][Y][Z];
		};
	}
}