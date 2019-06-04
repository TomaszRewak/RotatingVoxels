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
	}

	static class DiscreteEdgesExtension
	{
		public static bool CrossesZeroAt(this float[,,] voxelSpace, in DiscreteEdge edge)
		{
			return voxelSpace.At(edge.Begin) * voxelSpace.At(edge.End) <= 0;
		}

		public static Vertex CrossingPoint(this float[,,] voxelSpace, in DiscreteEdge edge)
		{

		}
	}
}
