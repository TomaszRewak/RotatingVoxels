using RotatingVoxels.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.VoxelSpace
{
	struct VoxelCell
	{
		public Vertex NearestIntersection;
		public Vector Normal;

		public VoxelCell(Vertex nearestIntersection, Vector normal)
		{
			NearestIntersection = nearestIntersection;
			Normal = normal;
		}
	}
}
