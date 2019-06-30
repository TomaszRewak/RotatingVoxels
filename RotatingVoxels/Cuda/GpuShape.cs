using Alea;
using RotatingVoxels.VoxelSpace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Cuda
{
	struct GpuShapeInfo
	{
		public DiscreteBounds Bounds;
		public VoxelCell[] Voxels;
	}

	class GpuShape
	{
		public GpuShapeInfo Shape { get; }

		public GpuShape(VoxelCell[,,] shape)
		{
			Shape = new GpuShapeInfo
			{
				Bounds = DiscreteBounds.Of(shape),
				Voxels = Gpu.Default.Allocate(Flatten(shape))
			};
		}

		private static VoxelCell[] Flatten(VoxelCell[,,] shape)
		{
			var bounds = DiscreteBounds.Of(shape);

			var flatShape = new VoxelCell[bounds.Length];
			foreach (var coordinates in bounds)
				flatShape[bounds.Index(coordinates)] = shape.At(coordinates);

			return flatShape;
		}
	}
}
