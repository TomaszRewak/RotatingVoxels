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
	}
}
