using RotatingVoxels.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.VoxelSpace
{
	struct DiscreteCoordinates
	{
		public int X;
		public int Y;
		public int Z;

		public static DiscreteCoordinates At(int x, int y, int z)
		{
			return new DiscreteCoordinates
			{
				X = x,
				Y = y,
				Z = z
			};
		}

		public static DiscreteCoordinates Floor(in Vertex point)
		{
			return new DiscreteCoordinates
			{
				X = (int)Math.Floor(point.X),
				Y = (int)Math.Floor(point.Y),
				Z = (int)Math.Floor(point.Z)
			};
		}

		public DiscreteCoordinates Move(int x, int y, int z)
		{
			return new DiscreteCoordinates
			{
				X = X + x,
				Y = Y + y,
				Z = Z + z
			};
		}

		public Vertex AsVertex()
		{
			return Vertex.At(X, Y, Z);
		}
	}

	static class DiscreteCoordinatesExtension
	{
		public static ref VoxelCell At(this VoxelCell[,,] voxelSpace, in DiscreteCoordinates coordinates)
		{
			return ref voxelSpace[coordinates.X, coordinates.Y, coordinates.Z];
		}
	}
}
