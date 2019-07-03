using OpenGL;
using RotatingVoxels.Cuda;
using RotatingVoxels.Geometry;
using RotatingVoxels.Resources.Models;
using RotatingVoxels.Resources.Shaders;
using RotatingVoxels.VoxelSpace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Scene
{
	class BunnyScene : IScene
	{
		private GpuSpace _gpuSpace;
		private GpuShape _gpuShape;

		private ShadingProgram _program;
		private Box _box;

		public void Draw(float width, float height, TimeSpan time)
		{
			float progress = time.Ticks / 10_000_000f;

			var worldTransformation = 
				Matrix4x4f.Perspective(40, 1f * width / height, 0.001f, 100000f) *
				Matrix4x4f.LookAt(new Vertex3f(0.2f, -0.5f * (float)Math.Sin(0.5), -1), new Vertex3f(0, 0, 0), new Vertex3f(0, -1, 0));

			using (_program.Use())
			{
				using (var context = _gpuSpace.UseBuffer())
				{
					var transformation = Matrix4x4f.Translated(progress * 2f, 0, 0) /** Matrix4x4f.RotatedZ(progress * 2)*/;

					VoxelKernel.Clear(context.Space);
					VoxelKernel.Sample(_gpuShape.Shape, context.Space, Matrix.From(transformation), maxDistance: 2);
					VoxelKernel.Normalize(context.Space, revert: true);
				}

				using (var context = _gpuSpace.UseTexture())
				{
					_program.Transformation = worldTransformation;
					_program.Weights = context.Texture;
					_program.Bounds = _gpuSpace.Bounds;

					_box.Draw(_gpuSpace.Bounds.Length);
				}
			}
		}

		public void Initialize()
		{
			_box = new Box();
			_gpuSpace = new GpuSpace(DiscreteBounds.OfSize(30, 30, 30));
			_program = new ShadingProgram();
		}

		public void Load()
		{
			_gpuShape = ShapeLoader.LoadShape("./Examples/bunny.stl", DiscreteBounds.OfSize(30, 30, 30), 5);
		}
	}
}
