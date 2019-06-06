using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Geometry
{
	struct Bounds
	{
		public float MinX;
		public float MinY;
		public float MinZ;

		public float MaxX;
		public float MaxY;
		public float MaxZ;

		public Bounds(float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
		{
			MinX = minX;
			MinY = minY;
			MinZ = minZ;

			MaxX = maxX;
			MaxY = maxY;
			MaxZ = maxZ;
		}

		public Bounds(in Face face)
		{
			MinX = Math.Min(Math.Min(face.A.Point.X, face.B.Point.X), face.C.Point.X);
			MinY = Math.Min(Math.Min(face.A.Point.Y, face.B.Point.Y), face.C.Point.Y);
			MinZ = Math.Min(Math.Min(face.A.Point.Z, face.B.Point.Z), face.C.Point.Z);

			MaxX = Math.Max(Math.Max(face.A.Point.X, face.B.Point.X), face.C.Point.X);
			MaxY = Math.Max(Math.Max(face.A.Point.Y, face.B.Point.Y), face.C.Point.Y);
			MaxZ = Math.Max(Math.Max(face.A.Point.Z, face.B.Point.Z), face.C.Point.Z);
		}

		public static Bounds Min => new Bounds
		{
			MinX = float.MaxValue,
			MinY = float.MaxValue,
			MinZ = float.MaxValue,

			MaxX = float.MinValue,
			MaxY = float.MinValue,
			MaxZ = float.MinValue
		};

		public static Bounds operator +(in Bounds boundsA, in Bounds boundsB)
		{
			return new Bounds
			{
				MinX = Math.Min(boundsA.MinX, boundsB.MinX),
				MinY = Math.Min(boundsA.MinY, boundsB.MinY),
				MinZ = Math.Min(boundsA.MinZ, boundsB.MinZ),

				MaxX = Math.Max(boundsA.MaxX, boundsB.MaxX),
				MaxY = Math.Max(boundsA.MaxY, boundsB.MaxY),
				MaxZ = Math.Max(boundsA.MaxZ, boundsB.MaxZ)
			};
		}
	}
}
