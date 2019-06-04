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
	}
}
