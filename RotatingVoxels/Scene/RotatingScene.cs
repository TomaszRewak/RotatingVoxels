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
	class RotatingScene : IScene
	{
		private GpuSpace _gpuSpace;
		private GpuShape _gpuShape;

		private ShadingProgram _program;
		private Box _box;

		public void Draw(float width, float height, TimeSpan time)
		{
			float progress = time.Ticks / 10_000_000f;

			using (_program.Use())
			{
				using (var context = _gpuSpace.UseBuffer())
				{
					var transformation = Matrix4x4f.Translated((float)Math.Sin(progress * 0.01) * 400f, 0, 0) * Matrix4x4f.RotatedZ(progress * 15f) * Matrix4x4f.RotatedY(progress * 4f);

					VoxelKernel.Clear(context.Space);
					VoxelKernel.Sample(_gpuShape.Shape, context.Space, Matrix.From(transformation), maxDistance: 5, revert: true);
					VoxelKernel.Normalize(context.Space);
				}

				using (var context = _gpuSpace.UseTexture())
				{
					_program.Transformation = GetWorldTransformation(width, height, progress);
					_program.Weights = context.Texture;
					_program.Bounds = _gpuSpace.Bounds;

					_box.Draw(_gpuSpace.Bounds.Length);
				}
			}
		}

		private Matrix4x4f GetWorldTransformation(float width, float height, float progress)
		{
			return 
				Matrix4x4f.Perspective(40, 1f * width / height, 0.001f, 100000f) *
				Matrix4x4f.LookAt(new Vertex3f((float)Math.Sin(progress * 0.08), -0.4f, (float)Math.Cos(progress * 0.08)), new Vertex3f(0, 0, 0), new Vertex3f(0, -1, 0));
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
