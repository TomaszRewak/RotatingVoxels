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
			MinX = Math.Min(Math.Min(face.A.X, face.B.X), face.C.X);
			MinY = Math.Min(Math.Min(face.A.Y, face.B.Y), face.C.Y);
			MinZ = Math.Min(Math.Min(face.A.Z, face.B.Z), face.C.Z);

			MaxX = Math.Max(Math.Max(face.A.X, face.B.X), face.C.X);
			MaxY = Math.Max(Math.Max(face.A.Y, face.B.Y), face.C.Y);
			MaxZ = Math.Max(Math.Max(face.A.Z, face.B.Z), face.C.Z);
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
