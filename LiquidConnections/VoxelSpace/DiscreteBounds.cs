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

		public DiscreteBounds(VoxelCell[,,] voxelSpace)
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

			MaxX = (int)Math.Floor(bounds.MaxX);
			MaxY = (int)Math.Floor(bounds.MaxY);
			MaxZ = (int)Math.Floor(bounds.MaxZ);
		}

		public DiscreteBounds Clip(in DiscreteBounds bounds)
		{
			return new DiscreteBounds
			{
				MinX = Math.Max(MinX, bounds.MinX),
				MinY = Math.Max(MinY, bounds.MinY),
				MinZ = Math.Max(MinZ, bounds.MinZ),

				MaxX = Math.Min(MaxX, bounds.MaxX),
				MaxY = Math.Min(MaxY, bounds.MaxY),
				MaxZ = Math.Min(MaxZ, bounds.MaxZ)
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
				MinX = MinX + minXOffset,
				MinY = MinY + minYOffset,
				MinZ = MinZ + minZOffset,

				MaxX = MaxX + maxXOffset,
				MaxY = MaxY + maxYOffset,
				MaxZ = MaxZ + maxZOffset
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
