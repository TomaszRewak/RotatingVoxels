using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Geometry
{
	struct Matrix
	{
		public float X1;
		public float Y1;
		public float Z1;
		public float W1;

		public float X2;
		public float Y2;
		public float Z2;
		public float W2;

		public float X3;
		public float Y3;
		public float Z3;
		public float W3;

		public float X4;
		public float Y4;
		public float Z4;
		public float W4;

		public static Matrix From(OpenGL.Matrix4x4f matrix)
		{
			return new Matrix
			{
				X1 = matrix.Row0.x,
				Y1 = matrix.Row0.y,
				Z1 = matrix.Row0.z,
				W1 = matrix.Row0.w,

				X2 = matrix.Row1.x,
				Y2 = matrix.Row1.y,
				Z2 = matrix.Row1.z,
				W2 = matrix.Row1.w,

				X3 = matrix.Row2.x,
				Y3 = matrix.Row2.y,
				Z3 = matrix.Row2.z,
				W3 = matrix.Row2.w,

				X4 = matrix.Row3.x,
				Y4 = matrix.Row3.y,
				Z4 = matrix.Row3.z,
				W4 = matrix.Row3.w
			};
		}

		public static Vertex operator*(in Matrix matrix, in Vertex vertex)
		{
			float w = matrix.X4 * vertex.X + matrix.Y4 * vertex.Y + matrix.Z4 * vertex.Z + matrix.W4;

			return new Vertex
			{
				X = (matrix.X1 * vertex.X + matrix.Y1 * vertex.Y + matrix.Z1 * vertex.Z + matrix.W1) / w,
				Y = (matrix.X2 * vertex.X + matrix.Y2 * vertex.Y + matrix.Z2 * vertex.Z + matrix.W2) / w,
				Z = (matrix.X3 * vertex.X + matrix.Y3 * vertex.Y + matrix.Z3 * vertex.Z + matrix.W3) / w,
			};
		}
	}
}
