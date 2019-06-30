using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Geometry
{
	struct FaceVertex
	{
		public Vertex Point;
		public Vector Normal;

		public FaceVertex(Vertex point, Vector normalVector)
		{
			Point = point;
			Normal = normalVector;
		}
	}
}
