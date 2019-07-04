using OpenGL;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Resources.Models
{
	class Box : IModel
	{
		private struct BoxContext : IDisposable
		{
			public void Dispose()
			{
				Gl.DisableVertexAttribArray(0);
				Gl.DisableVertexAttribArray(1);
			}
		}

		private uint colorDataBuffer;
		private static float[] colorDataValues = {
			0.1f, 0.2f, 0.3f,
			0.1f, 0.2f, 0.3f,
			0.1f, 0.2f, 0.3f,
			0.1f, 0.2f, 0.3f,
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
		};

		private uint vertexDataBuffer;
		private static float[] vertexDataValues = {
			-1.0f, -1.0f, +1.0f,
			+1.0f, -1.0f, +1.0f,
			+1.0f, +1.0f, +1.0f,
			-1.0f, +1.0f, +1.0f,
			-1.0f, -1.0f, -1.0f,
			+1.0f, -1.0f, -1.0f,
			+1.0f, +1.0f, -1.0f,
			-1.0f, +1.0f, -1.0f
		};

		private uint indexDataBuffer;
		private static ushort[] indexDataValues = {
			0, 1, 2,
			2, 3, 0,
			1, 5, 6,
			6, 2, 1,
			7, 6, 5,
			5, 4, 7,
			4, 0, 3,
			3, 7, 4,
			4, 5, 1,
			1, 0, 4,
			3, 2, 6,
			6, 7, 3
		};

		public Box()
		{
			vertexDataBuffer = Gl.GenBuffer();
			Gl.BindBuffer(BufferTarget.ArrayBuffer, vertexDataBuffer);
			Gl.BufferData(BufferTarget.ArrayBuffer, sizeof(float) * (uint)vertexDataValues.Length, vertexDataValues, BufferUsage.StaticDraw);

			colorDataBuffer = Gl.GenBuffer();
			Gl.BindBuffer(BufferTarget.ArrayBuffer, colorDataBuffer);
			Gl.BufferData(BufferTarget.ArrayBuffer, sizeof(float) * (uint)colorDataValues.Length, colorDataValues, BufferUsage.StaticDraw);

			indexDataBuffer = Gl.GenBuffer();
			Gl.BindBuffer(BufferTarget.ElementArrayBuffer, indexDataBuffer);
			Gl.BufferData(BufferTarget.ElementArrayBuffer, sizeof(ushort) * (uint)indexDataValues.Length, indexDataValues, BufferUsage.StaticDraw);
		}

		public void Dispose()
		{
			Gl.DeleteBuffers(vertexDataBuffer);
			Gl.DeleteBuffers(colorDataBuffer);
			Gl.DeleteBuffers(indexDataBuffer);
		}

		private IDisposable Bind()
		{
			Gl.EnableVertexAttribArray(0);
			Gl.BindBuffer(BufferTarget.ArrayBuffer, vertexDataBuffer);
			Gl.VertexAttribPointer(0, 3, VertexAttribType.Float, false, 0, IntPtr.Zero);

			Gl.EnableVertexAttribArray(1);
			Gl.BindBuffer(BufferTarget.ArrayBuffer, colorDataBuffer);
			Gl.VertexAttribPointer(1, 3, VertexAttribType.Float, false, 0, IntPtr.Zero);

			Gl.BindBuffer(BufferTarget.ElementArrayBuffer, indexDataBuffer);

			return new BoxContext();
		}

		public void Draw(int times)
		{
			using (Bind())
				Gl.DrawElementsInstanced(PrimitiveType.Triangles, indexDataValues.Length, DrawElementsType.UnsignedShort, IntPtr.Zero, times);
		}

		public void Draw()
		{
			Gl.DrawElements(PrimitiveType.Triangles, indexDataValues.Length, DrawElementsType.UnsignedShort, IntPtr.Zero);
		}
	}
}
