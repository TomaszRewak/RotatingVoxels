using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Geometry
{
	struct Vertex
	{
		public float X;
		public float Y;
		public float Z;

		public Vertex(float x, float y, float z)
		{
			X = x;
			Y = y;
			Z = z;
		}

		public static Vertex operator +(in Vertex vertex, in Vector vector)
		{
			return new Vertex(vertex.X + vector.X, vertex.X + vector.Y, vertex.Z + vector.Z);
		}

		public static Vertex operator -(in Vertex vertex, in Vector vector)
		{
			return new Vertex(vertex.X - vector.X, vertex.Y - vector.Y, vertex.Z - vector.Z);
		}
	}
}
