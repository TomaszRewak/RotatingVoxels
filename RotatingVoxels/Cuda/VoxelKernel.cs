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
		private static LaunchParam _lunchParams = new LaunchParam(32, 256);

		public static void Sample(GpuShapeInfo shape, GpuSpaceInfo space, Matrix transformation, float maxDistance, bool revert)
		{
			Gpu.Default.Launch(SampleKernel, _lunchParams, shape, space, transformation, maxDistance, revert);
			Gpu.Default.Synchronize();
		}

		private static void SampleKernel(GpuShapeInfo shape, GpuSpaceInfo space, Matrix transformation, float maxDistance, bool revert)
		{
			var start = space.Bounds.Length * (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x / gridDim.x;
			var end = space.Bounds.Length * (blockIdx.x * blockDim.x + threadIdx.x + 1) / blockDim.x / gridDim.x;

			var inversedTransformation = transformation.Inverse();

			for (int voxelIndex = start; voxelIndex < end; voxelIndex++)
			{
				var voxel = space.Voxels.Get(voxelIndex);

				var spaceCoordinates = space.Bounds.At(voxelIndex);
				var spacePosition = spaceCoordinates.AsVertex();
				var shapePosition = transformation * spacePosition;
				var shapeCoordinates = DiscreteCoordinates.Floor(shapePosition);

				for (int partIndex = 0; partIndex < 8; partIndex++)
				{
					var partCoordinates = shapeCoordinates.Move(partIndex % 2, (partIndex >> 1) % 2, (partIndex >> 2) % 2);
					var warpedPartCoordinates = shape.Bounds.Warp(partCoordinates);

					ref var cell = ref shape.Voxels[shape.Bounds.Index(warpedPartCoordinates)];
					var distance = Vector.Between(warpedPartCoordinates.AsVertex(), cell.NearestIntersection).Length;
					var normal = Vector.Between(
						spacePosition,
						inversedTransformation * (shapePosition + cell.Normal)
					).Normalize() * (revert ? -1 : 1);

					var weight = DeviceFunction.Max(0f, 1f - distance / maxDistance);
					var factor =
						(1 - DeviceFunction.Abs(partCoordinates.X - shapePosition.X)) *
						(1 - DeviceFunction.Abs(partCoordinates.Y - shapePosition.Y)) *
						(1 - DeviceFunction.Abs(partCoordinates.Z - shapePosition.Z));

					voxel.Weight += weight * factor;
					voxel.Normal += normal * factor * weight;
				}

				space.Voxels.Set(voxelIndex, voxel);
			}
		}

		public static void Clear(GpuSpaceInfo space)
		{
			Gpu.Default.Launch(ClearKernel, _lunchParams, space);
			Gpu.Default.Synchronize();
		}

		private static void ClearKernel(GpuSpaceInfo space)
		{
			var start = space.Bounds.Length * (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x / gridDim.x;
			var end = space.Bounds.Length * (blockIdx.x * blockDim.x + threadIdx.x + 1) / blockDim.x / gridDim.x;

			for (int i = start; i < end; i++)
				space.Voxels.Set(i, new VoxelFace());
		}

		public static void Normalize(GpuSpaceInfo space)
		{
			Gpu.Default.Launch(NormalizeKernel, _lunchParams, space);
			Gpu.Default.Synchronize();
		}

		private static void NormalizeKernel(GpuSpaceInfo space)
		{
			var start = space.Bounds.Length * (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x / gridDim.x;
			var end = space.Bounds.Length * (blockIdx.x * blockDim.x + threadIdx.x + 1) / blockDim.x / gridDim.x;

			for (int i = start; i < end; i++)
			{
				var value = space.Voxels.Get(i);

				space.Voxels.Set(i, new VoxelFace
				{
					Weight = DeviceFunction.Min(1, value.Weight),
					Normal = value.Normal.Normalize()
				});
			}
		}
	}
}
