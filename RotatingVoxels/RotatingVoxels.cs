﻿using Alea;
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
//using OpenGL;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using RotatingVoxels.Cuda;

namespace RotatingVoxels
{
	public struct TestType
	{
		public int A;
		public int B;
	}

	static class RotatingVoxels
	{
		private static int Add(in TestType a, in TestType b)
		{
			return a.A + b.B;
		}

		private static void Kernel(int[] result, TestType[] arg1, TestType[] arg2)
		{
			var start = blockIdx.x * blockDim.x + threadIdx.x;
			var stride = gridDim.x * blockDim.x;

			for (var i = start; i < result.Length; i += stride)
				for (int j = 0; j < arg1.Length; j++)
					result[i] = Add(arg1[i], arg2[j]);
		}

		[GpuManaged]
		public static int[] Run(int length)
		{
			var gpu = Gpu.Default;
			var lp = new LaunchParam(16, 256);
			var arg1 = Enumerable.Range(0, length).Select(e => new TestType { A = e, B = e }).ToArray();
			var arg2 = Enumerable.Range(0, length).Select(e => new TestType { A = e, B = e }).ToArray();
			var result = new int[length];

			gpu.Launch(Kernel, lp, result, arg1, arg2);

			return result;
		}

		private static void KernelCPU(int[] result, int[] arg1, int[] arg2)
		{
			for (var i = 0; i < result.Length; i++)
				for (int j = 0; j < arg1.Length; j++)
					result[i] = arg1[i] + arg2[j];
		}

		public static int[] RunCPU(int length)
		{
			var arg1 = Enumerable.Range(0, length).ToArray();
			var arg2 = Enumerable.Range(0, length).ToArray();
			var result = new int[length];

			KernelCPU(result, arg1, arg2);

			return result;
		}

		static VoxelCell[,,] voxelizedBunny;
		static Face[] bunny;
		static GpuShape gpuBunny;
		static GpuSpace gpuSpace;

		static void Main(string[] args)
		{
			bunny = StlReader.LoadShape("./Examples/bunny.stl");

			Stopwatch stopwatch = Stopwatch.StartNew();

			voxelizedBunny = VoxelSpaceBuilder.Build(ShapeNormalizer.NormalizeShape(bunny, new Bounds(5, 5, 5, 35, 35, 35)), DiscreteBounds.OfSize(40, 40, 40));

			stopwatch.Stop();
			Console.WriteLine($"Pricessing took: {stopwatch.ElapsedMilliseconds}ms");

			gpuBunny = new GpuShape(voxelizedBunny);

			using (NativeWindow nativeWindow = NativeWindow.Create())
			{
				nativeWindow.DepthBits = 24;
				nativeWindow.Create(100, 100, 1000, 1000, NativeWindowStyle.Overlapped);
				nativeWindow.Show();
				nativeWindow.Render += Render;
				nativeWindow.Resize += Resize;
				Initialize();
				Resize(nativeWindow, null);
				nativeWindow.Run();
			}
		}

		private static void Resize(object sender, EventArgs e)
		{
			var window = sender as NativeWindow;
			var aspect = (float)window.Width / (float)window.Height;

			Gl.Viewport(0, 0, (int)window.Width, (int)window.Height);

			Gl.MatrixMode(MatrixMode.Projection);
			Gl.LoadIdentity();

			Matrix4x4f perspective = Matrix4x4f.Perspective(45.0f, aspect, 0.1f, 100.0f);
			Gl.LoadMatrixf(perspective);
		}

		static IModel model;
		static ShadingProgram program;

		static int iteration = 0;
		static int fps = 0;
		static Stopwatch stopwatch = new Stopwatch();
		private static void Render(object sender, NativeWindowEventArgs e)
		{
			fps++;
			if (stopwatch.ElapsedMilliseconds >= 1000)
			{
				Console.WriteLine(fps * 1000f / stopwatch.ElapsedMilliseconds);
				stopwatch.Restart();
				fps = 0;
			}

			Gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

			Gl.MatrixMode(MatrixMode.Modelview);

			using(program.Use())
			{
				using (var context = gpuSpace.UseBuffer())
				{
					VoxelKernel.Clear(context.Space);
					VoxelKernel.Sample(gpuBunny.Shape, context.Space, iteration * .1f);
				}

				using (var context = gpuSpace.UseTexture())
				{
					program.Transformation = Matrix4x4f.Perspective(60, 1, 0.001f, 100000f) * Matrix4x4f.LookAt(new Vertex3f(0.2f, -0.5f * (float)Math.Sin(iteration * 0.005), -1), new Vertex3f(0, 0, 0), new Vertex3f(0, -1, 0));
					program.Weights = context.Texture;

					model.Draw(40 * 40 * 40);
				}
			}

			Gl.LoadIdentity();

			Gl.Translate(0.0f, 0.0f, -7.0f);

			Gl.LoadIdentity();

			iteration++;
		}

		private static void Initialize()
		{
			Gl.Light(LightName.Light0, LightParameter.Diffuse, new[] { 0.7f, 0.7f, 0.7f, 1.0f });
			Gl.Light(LightName.Light0, LightParameter.Position, new[] { 1.0f, 1.0f, 13.0f, 0.0f });
			Gl.Enable(EnableCap.ColorMaterial);

			Gl.Enable(EnableCap.Light0);
			Gl.Enable(EnableCap.Lighting);

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

			model = new Box();
			gpuSpace = new GpuSpace(DiscreteBounds.OfSize(40, 40, 40));

			program = new ShadingProgram();

			stopwatch.Start();
		}
	}
}
