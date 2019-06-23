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

		public static Vertex At(float x, float y, float z)
		{
			return new Vertex
			{
				X = x,
				Y = y,
				Z = z
			};
		}

		public static Vertex MaxValue => At(float.MaxValue, float.MaxValue, float.MaxValue);

		public static Vertex operator +(in Vertex vertex, in Vector vector)
		{
			return At(vertex.X + vector.X, vertex.Y + vector.Y, vertex.Z + vector.Z);
		}

		public static Vertex operator -(in Vertex vertex, in Vector vector)
		{
			return At(vertex.X - vector.X, vertex.Y - vector.Y, vertex.Z - vector.Z);
		}

		public static bool operator ==(in Vertex first, in Vertex second)
		{
			return first.X == second.X && first.Y == second.Y && first.Z == second.Z;
		}

		public static bool operator !=(in Vertex first, in Vertex second)
		{
			return !(first == second);
		}
	}
}
