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

		public Matrix Inverse()
		{
			var i1 = Z3 * W4 - W3 * Z4;
			var i2 = Y3 * W4 - W3 * Y4;
			var i3 = Y3 * Z4 - Z3 * Y4;
			var i4 = X3 * W4 - W3 * X4;
			var i5 = X3 * Z4 - Z3 * X4;
			var i6 = X3 * Y4 - Y3 * X4;
			var i7 = Z2 * W4 - W2 * Z4;
			var i8 = Y2 * W4 - W2 * Y4;
			var i9 = Y2 * Z4 - Z2 * Y4;
			var i10 = Z2 * W3 - W2 * Z3;
			var i11 = Y2 * W3 - W2 * Y3;
			var i12 = Y2 * Z3 - Z2 * Y3;
			var i13 = X2 * W4 - W2 * X4;
			var i14 = X2 * Z4 - Z2 * X4;
			var i15 = X2 * W3 - W2 * X3;
			var i16 = X2 * Z3 - Z2 * X3;
			var i17 = X2 * Y4 - Y2 * X4;
			var i18 = X2 * Y3 - Y2 * X3;

			var det = X1 * (Y2 * i1 - Z2 * i2 + W2 * i3)
				- Y1 * (X2 * i1 - Z2 * i4 + W2 * i5)
				+ Z1 * (X2 * i2 - Y2 * i4 + W2 * i6)
				- W1 * (X2 * i3 - Y2 * i5 + Z2 * i6);
			det = 1 / det;

			return new Matrix()
			{
				X1 = det * (Y2 * i1 - Z2 * i2 + W2 * i3),
				Y1 = det * -(Y1 * i1 - Z1 * i2 + W1 * i3),
				Z1 = det * (Y1 * i7 - Z1 * i8 + W1 * i9),
				W1 = det * -(Y1 * i10 - Z1 * i11 + W1 * i12),
				X2 = det * -(X2 * i1 - Z2 * i4 + W2 * i5),
				Y2 = det * (X1 * i1 - Z1 * i4 + W1 * i5),
				Z2 = det * -(X1 * i7 - Z1 * i13 + W1 * i14),
				W2 = det * (X1 * i10 - Z1 * i15 + W1 * i16),
				X3 = det * (X2 * i2 - Y2 * i4 + W2 * i6),
				Y3 = det * -(X1 * i2 - Y1 * i4 + W1 * i6),
				Z3 = det * (X1 * i8 - Y1 * i13 + W1 * i17),
				W3 = det * -(X1 * i11 - Y1 * i15 + W1 * i18),
				X4 = det * -(X2 * i3 - Y2 * i5 + Z2 * i6),
				Y4 = det * (X1 * i3 - Y1 * i5 + Z1 * i6),
				Z4 = det * -(X1 * i9 - Y1 * i14 + Z1 * i17),
				W4 = det * (X1 * i12 - Y1 * i16 + Z1 * i18),
			};
		}

		public static Vertex operator *(in Matrix matrix, in Vertex vertex)
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
