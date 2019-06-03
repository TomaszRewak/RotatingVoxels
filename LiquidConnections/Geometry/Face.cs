using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Geometry
{
	struct Face
	{
		public Vertex A;
		public Vertex B;
		public Vertex C;

		public Vector Normal;

		public Face(Vertex a, Vertex b, Vertex c, Vector normal)
		{
			A = a;
			B = b;
			C = c;

			Normal = normal;
		}

		public float MinX => Math.Min(Math.Min(A.X, B.X), C.X);
		public float MinY => Math.Min(Math.Min(A.Y, B.Y), C.Y);
		public float MinZ => Math.Min(Math.Min(A.Z, B.Z), C.Z);

		public float MaxX => Math.Max(Math.Max(A.X, B.X), C.X);
		public float MaxY => Math.Max(Math.Max(A.Y, B.Y), C.Y);
		public float MaxZ => Math.Max(Math.Max(A.Z, B.Z), C.Z);
	}
}
