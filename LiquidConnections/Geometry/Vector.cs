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

		public float Length => (float)Math.Sqrt(X * X + Y * Y + Z * Z);

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

		public Vector Normalize()
		{
			var length = Length;

			return new Vector(X / Length, Y / Length, Z / Length);
		}

		public static Vector operator -(in Vector vectorA)
		{
			return new Vector(-vectorA.X, -vectorA.Y, -vectorA.Z);
		}

		public static Vector operator +(in Vector vectorA, in Vector vectorB)
		{
			return new Vector(vectorA.X + vectorB.X, vectorA.Y + vectorB.Y, vectorA.Z + vectorB.Z);
		}

		public static Vector operator *(in Vector vectorA, in Vector vectorB)
		{
			return new Vector(vectorA.X * vectorB.X, vectorA.Y * vectorB.Y, vectorA.Z * vectorB.Z);
		}

		public static Vector operator /(in Vector vectorA, in Vector vectorB)
		{
			return new Vector(vectorA.X / vectorB.X, vectorA.Y / vectorB.Y, vectorA.Z / vectorB.Z);
		}

		public static Vector operator *(in Vector vector, float By)
		{
			return new Vector(vector.X * By, vector.Y * By, vector.Z * By);
		}

		public static Vector operator /(in Vector vector, float By)
		{
			return new Vector(vector.X / By, vector.Y / By, vector.Z / By);
		}
	}
}
