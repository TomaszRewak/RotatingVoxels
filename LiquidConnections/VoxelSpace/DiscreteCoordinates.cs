using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
{
	struct DiscreteCoordinates
	{
		public int X;
		public int Y;
		public int Z;

		public DiscreteCoordinates(int x, int y, int z)
		{
			X = x;
			Y = y;
			Z = z;
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
			return new Vertex(X, Y, Z);
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
