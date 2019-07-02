using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Geometry
{
	struct Vector
	{
		public float X;
		public float Y;
		public float Z;

		public static Vector To(float x, float y, float z)
		{
			return new Vector
			{
				X = x,
				Y = y,
				Z = z
			};
		}

		public static Vector Between(in Vertex begin, in Vertex end)
		{
			return new Vector
			{
				X = end.X - begin.X,
				Y = end.Y - begin.Y,
				Z = end.Z - begin.Z
			};
		}

		public float Length => (float)Math.Sqrt(X * X + Y * Y + Z * Z);

		public float DotProduct(in Vector second)
		{
			return X * second.X + Y * second.Y + Z * second.Z;
		}

		public Vector CrossProduct(in Vector second)
		{
			return To(
				Y * second.Z - Z * second.Y,
				Z * second.X - X * second.Z,
				X * second.Y - Y * second.X
			);
		}

		public Vector Normalize()
		{
			var length = Length;

			if (length != 0)
				return To(X / length, Y / length, Z / length);
			else
				return new Vector();
		}

		public static Vector operator -(in Vector vectorA)
		{
			return To(-vectorA.X, -vectorA.Y, -vectorA.Z);
		}

		public static Vector operator +(in Vector vectorA, in Vector vectorB)
		{
			return To(vectorA.X + vectorB.X, vectorA.Y + vectorB.Y, vectorA.Z + vectorB.Z);
		}

		public static Vector operator *(in Vector vectorA, in Vector vectorB)
		{
			return To(vectorA.X * vectorB.X, vectorA.Y * vectorB.Y, vectorA.Z * vectorB.Z);
		}

		public static Vector operator /(in Vector vectorA, in Vector vectorB)
		{
			return To(vectorA.X / vectorB.X, vectorA.Y / vectorB.Y, vectorA.Z / vectorB.Z);
		}

		public static Vector operator *(in Vector vector, float By)
		{
			return To(vector.X * By, vector.Y * By, vector.Z * By);
		}

		public static Vector operator /(in Vector vector, float By)
		{
			return To(vector.X / By, vector.Y / By, vector.Z / By);
		}
	}
}
