using Alea;
using Alea.CSharp;
using RotatingVoxels.Geometry;
using RotatingVoxels.Resources;
using RotatingVoxels.Resources.Models;
using RotatingVoxels.Resources.Shaders;
using RotatingVoxels.Shapes;
using RotatingVoxels.Stl;
using RotatingVoxels.VoxelSpace;
using OpenGL;
using OpenGL.CoreUI;
using System;
using System.Diagnostics;
using RotatingVoxels.Cuda;
using RotatingVoxels.Window;
using RotatingVoxels.Scene;

namespace RotatingVoxels
{
	static class RotatingVoxels
	{
		static FpsCounter fpsCounter = new FpsCounter();

		static GpuShape gpuShape;
		static GpuSpace gpuSpace;

		static NativeWindow window;
		static ShadingProgram program;
		static IModel cellModel;

		static void Main(string[] args)
		{
			gpuShape = ShapeLoader.LoadShape("./Examples/bunny.stl", DiscreteBounds.OfSize(40, 40, 40), 5);

			using (window = NativeWindow.Create())
			{
				InitializeWindow();
				InitializeOpenGl();

				cellModel = new Box();
				gpuSpace = new GpuSpace(DiscreteBounds.OfSize(40, 40, 40));
				program = new ShadingProgram();

				fpsCounter.Start();

				window.Show();
				window.Run();
			}
		}

		static int iteration = 0;
		private static void Render(object sender, NativeWindowEventArgs e)
		{
			fpsCounter.Tick();

			Gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

			using(program.Use())
			{
				using (var context = gpuSpace.UseBuffer())
				{
					var transformation = Matrix4x4f.Translated(iteration * 0.1f, 0, 0) * Matrix4x4f.RotatedZ(iteration * 0.1f);

					VoxelKernel.Clear(context.Space);
					VoxelKernel.Sample(gpuShape.Shape, context.Space, Matrix.From(transformation));
					VoxelKernel.Normalize(context.Space);
				}

				using (var context = gpuSpace.UseTexture())
				{
					program.Transformation = Matrix4x4f.Perspective(60, 1f * window.Width / window.Height, 0.001f, 100000f) * Matrix4x4f.LookAt(new Vertex3f(0.2f, -0.5f * (float)Math.Sin(iteration * 0.005), -1), new Vertex3f(0, 0, 0), new Vertex3f(0, -1, 0));
					program.Weights = context.Texture;

					cellModel.Draw(gpuSpace.Bounds.Length);
				}
			}

			iteration++;
		}

		private static void InitializeWindow()
		{
			window.DepthBits = 24;
			window.Create(100, 100, 1200, 800, NativeWindowStyle.None);
			window.Render += Render;
		}

		private static void InitializeOpenGl()
		{
			Gl.ClearColor(0.2f, 0.2f, 0.2f, 1.0f);
			Gl.ClearDepth(1.0f);
			Gl.Enable(EnableCap.DepthTest);
			Gl.DepthFunc(DepthFunction.Lequal);
			Gl.ShadeModel(ShadingModel.Smooth);
			Gl.Hint(HintTarget.PerspectiveCorrectionHint, HintMode.Nicest);
			Gl.Enable(EnableCap.Normalize);

			Gl.Enable(EnableCap.CullFace);
			Gl.CullFace(CullFaceMode.Back);
			Gl.FrontFace(FrontFaceDirection.Ccw);
		}
	}
}
