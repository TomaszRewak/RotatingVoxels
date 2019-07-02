using Alea;
using OpenGL;
using RotatingVoxels.VoxelSpace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Cuda
{
	struct GpuSpaceInfo
	{
		public DiscreteBounds Bounds;
		public deviceptr<VoxelFace> Voxels;
	}

	class GpuSpace
	{
		private readonly uint _buffer;
		private readonly uint _texture;

		public DiscreteBounds Bounds { get; }

		public GpuSpace(DiscreteBounds bounds)
		{
			_buffer = Gl.GenBuffer();
			_texture = Gl.GenTexture();

			Bounds = bounds;

			Gl.BindBuffer(BufferTarget.TextureBuffer, _buffer);
			Gl.BufferData(BufferTarget.TextureBuffer, (uint)Marshal.SizeOf<VoxelFace>() * (uint)bounds.Length, null, BufferUsage.DynamicDraw);
		}

		public GpuSpaceBufferContext UseBuffer()
		{
			return new GpuSpaceBufferContext(_buffer, Bounds);
		}

		public GpuSpaceTextureContext UseTexture()
		{
			return new GpuSpaceTextureContext(_texture, _buffer);
		}
	}

	class GpuSpaceBufferContext : IDisposable
	{
		private readonly uint _buffer;
		private readonly deviceptr<VoxelFace> _devicePointer;
		private readonly DiscreteBounds _bounds;

		public GpuSpaceInfo Space => new GpuSpaceInfo { Bounds = _bounds, Voxels = _devicePointer };

		public GpuSpaceBufferContext(uint buffer, DiscreteBounds bounds)
		{
			CUDAInterop.cuGLRegisterBufferObject(buffer);

			_buffer = buffer;
			_devicePointer = new deviceptr<VoxelFace>(GetDevicePointer());
			_bounds = bounds;
		}

		public void Dispose()
		{
			CUDAInterop.cuGLUnmapBufferObject(_buffer);
			CUDAInterop.cuGLUnregisterBufferObject(_buffer);
		}

		private IntPtr GetDevicePointer()
		{
			IntPtr pointer, size;
			unsafe
			{
				CUDAInterop.cuSafeCall(CUDAInterop.cuGLMapBufferObject(&pointer, &size, _buffer));
			}
			return pointer;
		}
	}

	class GpuSpaceTextureContext : IDisposable
	{
		private readonly uint _buffer;

		public uint Texture { get; }

		public GpuSpaceTextureContext(uint texture, uint buffer)
		{
			Texture = texture;
			_buffer = buffer;

			Gl.ActiveTexture(TextureUnit.Texture1);
			Gl.BindTexture((TextureTarget)Gl.TEXTURE_BUFFER, Texture);
			Gl.TexBuffer((TextureTarget)Gl.TEXTURE_BUFFER, InternalFormat.Rgba32f, _buffer);
		}

		public void Dispose()
		{ }
	}
}
