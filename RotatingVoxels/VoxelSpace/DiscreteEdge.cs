using RotatingVoxels.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.VoxelSpace
{
	struct DiscreteEdge
	{
		public DiscreteCoordinates Begin;
		public DiscreteCoordinates End;

		public DiscreteEdge(in DiscreteCoordinates begin, in DiscreteCoordinates end)
		{
			Begin = begin;
			End = end;
		}

		public Vector AsVector()
		{
			return Vector.Between(Begin.AsVertex(), End.AsVertex());
		}
	}
}
