using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Geometry
{
	struct Vector
	{
		public float X;
		public float Y;
		public float Z;

		public Vector(float x, float y, float z)
		{
			X = x;
			Y = y;
			Z = z;
		}

		public Vector(in Vertex begin, in Vertex end)
		{
			X = end.X - begin.X;
			Y = end.Y - begin.Y;
			Z = end.Z - begin.Z;
		}

		public float DotProduct(in Vector second)
		{
			return X * second.X + Y * second.Y + Z * second.Z;
		}

		public Vector CrossProduct(in Vector second)
		{
			return new Vector(
				Y * second.Z - Z * second.Y,
				Z * second.X - X * second.Z,
				X * second.Y - Y * second.X
			);
		}

		public static Vector operator *(in Vector vector, float By)
		{
			return new Vector(vector.X * By, vector.Y * By, vector.Z * By);
		}
	}
}
