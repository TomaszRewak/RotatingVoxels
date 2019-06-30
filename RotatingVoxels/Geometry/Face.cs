using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Geometry
{
	struct Face
	{
		public FaceVertex A;
		public FaceVertex B;
		public FaceVertex C;

		public Face(FaceVertex a, FaceVertex b, FaceVertex c)
		{
			A = a;
			B = b;
			C = c;
		}

		public Vector Normal(in Vertex vertex)
		{
			var areaABC = Area(A.Point, B.Point, C.Point);
			var areaABP = Area(A.Point, B.Point, vertex);
			var areaACP = Area(A.Point, C.Point, vertex);

			var c = areaABP / areaABC;
			var b = areaACP / areaABC;
			var a = 1 - b - c;

			return (A.Normal * a + B.Normal * b + C.Normal * c).Normalize();
		}

		private float Area(in Vertex a, in Vertex b, in Vertex c)
		{
			var vectorAB = Vector.Between(a, b);
			var vectorAC = Vector.Between(a, c);

			return vectorAB.CrossProduct(vectorAC).Length / 2;
		}
	}
}
