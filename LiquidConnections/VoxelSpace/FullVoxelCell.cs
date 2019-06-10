using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
{
	struct FullVoxelCell
	{
		public Vertex NearestIntersection;
		public Vector Normal;

		public FullVoxelCell(Vertex nearestIntersection, Vector normal)
		{
			NearestIntersection = nearestIntersection;
			Normal = normal;
		}
	}
}
