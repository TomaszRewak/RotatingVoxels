using RotatingVoxels.Cuda;
using RotatingVoxels.Geometry;
using RotatingVoxels.Shapes;
using RotatingVoxels.Stl;
using RotatingVoxels.VoxelSpace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Scene
{
	static class ShapeLoader
	{
		public static GpuShape LoadShape(string path, DiscreteBounds size, int borderSize)
		{
			var normalizedBounds = new Bounds
			{
				MinX = size.MinX + borderSize,
				MinY = size.MinY + borderSize,
				MinZ = size.MinZ + borderSize,
				MaxX = size.MaxX - borderSize,
				MaxY = size.MaxY - borderSize,
				MaxZ = size.MaxZ - borderSize,
			};

			var stlShape = StlReader.LoadShape(path);
			var normalizedShape = ShapeNormalizer.NormalizeShape(stlShape, normalizedBounds);
			var voxelizedShape = VoxelSpaceBuilder.Build(normalizedShape, size);

			return new GpuShape(voxelizedShape);
		}
	}
}
