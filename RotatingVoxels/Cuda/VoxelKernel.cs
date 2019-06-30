using Alea;
using Alea.CSharp;
using RotatingVoxels.Geometry;
using RotatingVoxels.VoxelSpace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Cuda
{
	class VoxelKernel
	{
		private static LaunchParam _lunchParams = new LaunchParam(1, 256);

		public static void Sample(GpuShapeInfo shape, GpuSpaceInfo space, float xOffset)
		{
			Gpu.Default.Launch(SampleKernel, _lunchParams, shape, space, xOffset * 0.2f);
		}

		private static void SampleKernel(GpuShapeInfo shape, GpuSpaceInfo space, float xOffset)
		{
			var length = 40 * 40 * 40;
			var start = length * (threadIdx.x) / blockDim.x;
			var end = length * (threadIdx.x + 1) / blockDim.x;
			int offset = (int)Math.Floor(xOffset);

			for (int i = start; i < end; i++)
			{
				int x = i % 40 + offset;
				int y = i / 40 % 40;
				int z = i / 40 / 40 % 40;

				var coordinates1 = DiscreteCoordinates.At(x % 40, y, z);
				var coordinates2 = DiscreteCoordinates.At((x + 1) % 40, y, z);
				var bounds = DiscreteBounds.OfSize(40, 40, 40);

				xOffset = xOffset - (float)Math.Floor(xOffset);

				ref var cell1 = ref shape.Voxels[bounds.Index(coordinates1)];
				var distance1 = Vector.Between(coordinates1.AsVertex(), cell1.NearestIntersection).Length;

				ref var cell2 = ref shape.Voxels[bounds.Index(coordinates2)];
				var distance2 = Vector.Between(coordinates2.AsVertex(), cell2.NearestIntersection).Length;

				var weight1 = Math.Max(0f, 1f - distance1 / 4);
				var weight2 = Math.Max(0f, 1f - distance2 / 4);

				space.Voxels.Set(i, new VoxelFace
				{
					Weight = weight1 * (1 - xOffset) + weight2 * xOffset,
					Normal = cell1.Normal * (1 - xOffset) + cell2.Normal * xOffset
				});
			}
		}

		public static void Clear(GpuSpaceInfo space)
		{
			Gpu.Default.Launch(ClearKernel, _lunchParams, space);
		}

		private static void ClearKernel(GpuSpaceInfo space)
		{
			var start = space.Bounds.Length * (threadIdx.x) / blockDim.x;
			var end = space.Bounds.Length * (threadIdx.x + 1) / blockDim.x;

			for (int i = start; i < end; i++)
				space.Voxels.Set(i, new VoxelFace { Weight = 0 });
		}
	}
}
