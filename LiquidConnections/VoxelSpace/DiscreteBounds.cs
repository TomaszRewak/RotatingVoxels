using LiquidConnections.Geometry;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
{
	struct DiscreteBounds
	{
		public int MinX;
		public int MinY;
		public int MinZ;

		public int MaxX;
		public int MaxY;
		public int MaxZ;

		public DiscreteBounds(float[,,] voxelSpace)
		{
			MinX = 0;
			MinY = 0;
			MinZ = 0;

			MaxX = voxelSpace.GetLength(0) - 1;
			MaxY = voxelSpace.GetLength(1) - 1;
			MaxZ = voxelSpace.GetLength(2) - 1;
		}

		public DiscreteBounds(in Bounds bounds)
		{
			MinX = (int)Math.Ceiling(bounds.MinX);
			MinY = (int)Math.Ceiling(bounds.MinY);
			MinZ = (int)Math.Ceiling(bounds.MinZ);

			MaxX = (int)Math.Floor(bounds.MinX);
			MaxY = (int)Math.Floor(bounds.MinY);
			MaxZ = (int)Math.Floor(bounds.MinZ);
		}

		public DiscreteBounds Clip(in DiscreteBounds bounds)
		{
			return new DiscreteBounds
			{
				MinX = Math.Max(MinX, bounds.MinX),
				MinY = Math.Max(MinY, bounds.MinY),
				MinZ = Math.Max(MinZ, bounds.MinZ),

				MaxX = Math.Min(MinX, bounds.MinX),
				MaxY = Math.Min(MinY, bounds.MinY),
				MaxZ = Math.Min(MinZ, bounds.MinZ)
			};
		}

		public DiscreteBounds Offset(
			int minXOffset,
			int minYOffset,
			int minZOffset,
			int maxXOffset,
			int maxYOffset,
			int maxZOffset)
		{
			return new DiscreteBounds
			{
				MinX = MinX + maxXOffset,
				MinY = MinY + maxYOffset,
				MinZ = MinZ + maxZOffset,

				MaxX = MinX + minXOffset,
				MaxY = MinY + minYOffset,
				MaxZ = MinZ + minZOffset
			};
		}

		public bool Inside(in DiscreteCoordinates coordinates)
		{
			return
				coordinates.X >= MinX && coordinates.X <= MaxX &&
				coordinates.Y >= MinY && coordinates.Y <= MaxY &&
				coordinates.Z >= MinZ && coordinates.Z <= MaxZ;
		}

		public IEnumerator<DiscreteCoordinates> GetEnumerator()
		{
			for (int x = MinX; x <= MaxX; x++)
				for (int y = MinY; y <= MaxY; y++)
					for (int z = MinZ; z <= MaxZ; z++)
						yield return new DiscreteCoordinates(x, y, z);
		}
	}
}
