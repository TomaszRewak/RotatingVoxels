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
		public float Distance;
		public Vector Normal;

		public VoxelCell(float distance, Vector normal)
		{
			Distance = distance;
			Normal = normal;
		}
	}
}
