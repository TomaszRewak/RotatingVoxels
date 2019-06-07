using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
{
	struct VoxelCell
	{
		public Vertex NearestIntersection;
		public Vector Normal;
		public int Weight;

		public VoxelCell(Vertex nearestIntersection, Vector normal, int weight = 0)
		{
			NearestIntersection = nearestIntersection;
			Normal = normal;
			Weight = weight;
		}
	}
}
