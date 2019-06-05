using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
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
			return new Vector(Begin.AsVertex(), End.AsVertex());
		}
	}
}
