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

		public static void Sample(GpuShapeInfo shape, GpuSpaceInfo space, Matrix transformation)
		{
			Gpu.Default.Launch(SampleKernel, _lunchParams, shape, space, transformation);
		}

		private static void SampleKernel(GpuShapeInfo shape, GpuSpaceInfo space, Matrix transformation)
		{
			var start = space.Bounds.Length * (threadIdx.x) / blockDim.x;
			var end = space.Bounds.Length * (threadIdx.x + 1) / blockDim.x;

			for (int voxelIndex = start; voxelIndex < end; voxelIndex++)
			{
				var voxel = new VoxelFace();

				var spaceCoordinates = DiscreteCoordinates.At(voxelIndex % 40, voxelIndex / 40 % 40, voxelIndex / 40 / 40 % 40);
				var spacePosition = spaceCoordinates.AsVertex();
				var shapePosition = transformation * spacePosition;
				var shapeCoordinates = DiscreteCoordinates.Floor(shapePosition);

				for (int partIndex = 0; partIndex < 8; partIndex++)
				{
					var partCoordinates = shapeCoordinates.Move(partIndex % 2, (partIndex >> 1) % 2, (partIndex >> 2) % 2);
					var warpedPartCoordinates = shape.Bounds.Warp(partCoordinates);

					ref var cell = ref shape.Voxels[shape.Bounds.Index(warpedPartCoordinates)];
					var distance = Vector.Between(warpedPartCoordinates.AsVertex(), cell.NearestIntersection).Length;

					var weight = DeviceFunction.Max(0f, 1f - distance / 4);
					var factor = 
						(1 - DeviceFunction.Abs(partCoordinates.X - shapePosition.X)) * 
						(1 - DeviceFunction.Abs(partCoordinates.Y - shapePosition.Y)) * 
						(1 - DeviceFunction.Abs(partCoordinates.Z - shapePosition.Z));

					voxel.Weight += weight * factor;
					voxel.Normal += cell.Normal * weight * factor;
				}

				space.Voxels.Set(voxelIndex, voxel);
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
