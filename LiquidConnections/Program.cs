using Alea;
using Alea.CSharp;
using LiquidConnections.Geometry;
using LiquidConnections.Shapes;
using LiquidConnections.Stl;
using LiquidConnections.VoxelSpace;
using OpenGL;
using OpenGL.CoreUI;
using System;
//using OpenGL;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace LiquidConnections
{
	public struct TestType
	{
		public int A;
		public int B;
	}

	static class Program
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

		static Face[] bunny;

		static void Main(string[] args)
		{
			//Stopwatch stopwatch;

			//var l = 50000;

			//stopwatch = Stopwatch.StartNew();
			//Run(l);
			//stopwatch.Stop();
			//Console.WriteLine(stopwatch.ElapsedMilliseconds);

			//stopwatch = Stopwatch.StartNew();
			//RunCPU(l);
			//stopwatch.Stop();
			//Console.WriteLine(stopwatch.ElapsedMilliseconds);

			//stopwatch = Stopwatch.StartNew();
			//Run(l);
			//stopwatch.Stop();
			//Console.WriteLine(stopwatch.ElapsedMilliseconds);

			//stopwatch = Stopwatch.StartNew();
			//RunCPU(l);
			//stopwatch.Stop();
			//Console.WriteLine(stopwatch.ElapsedMilliseconds);

			bunny = StlReader.LoadShape("./Examples/bunny.stl");

			var voxelSpaceBuilder = new VoxelSpaceBuilder(20, 20, 20);
			voxelSpaceBuilder.Add(ShapeNormalizer.NormalizeShape(bunny, new Bounds(0, 0, 0, 20, 20, 20)));

			bunny = ShapeNormalizer.NormalizeShape(VoxelSpaceReader.GenerateShape(voxelSpaceBuilder.VoxelSpace), new Bounds(-2, -2, -2, 2, 2, 2));

			using (NativeWindow nativeWindow = NativeWindow.Create())
			{
				nativeWindow.DepthBits = 24;
				nativeWindow.Create(100, 100, 640, 480, NativeWindowStyle.Overlapped);
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

		static int offset = 0;
		private static void Render(object sender, NativeWindowEventArgs e)
		{
			Gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

			Gl.MatrixMode(MatrixMode.Modelview);

			Gl.LoadIdentity();
			Gl.Translate(Math.Cos(offset * 0.01) * 3, Math.Cos(offset * 0.023) * 3, -7.0f);
			offset++;

			Gl.Begin(PrimitiveType.Quads);

			Gl.Color3(0.0f, 1.0f, 0.0f);     // Green
			Gl.Vertex3(1.0f, 1.0f, -1.0f);
			Gl.Vertex3(-1.0f, 1.0f, -1.0f);
			Gl.Vertex3(-1.0f, 1.0f, 1.0f);
			Gl.Vertex3(1.0f, 1.0f, 1.0f);

			// Bottom face (y = -1.0f)
			Gl.Color3(1.0f, 0.5f, 0.0f);     // Orange
			Gl.Vertex3(1.0f, -1.0f, 1.0f);
			Gl.Vertex3(-1.0f, -1.0f, 1.0f);
			Gl.Vertex3(-1.0f, -1.0f, -1.0f);
			Gl.Vertex3(1.0f, -1.0f, -1.0f);

			// Front face  (z = 1.0f)
			Gl.Color3(1.0f, 0.0f, 0.0f);     // Red
			Gl.Vertex3(1.0f, 1.0f, 1.0f);
			Gl.Vertex3(-1.0f, 1.0f, 1.0f);
			Gl.Vertex3(-1.0f, -1.0f, 1.0f);
			Gl.Vertex3(1.0f, -1.0f, 1.0f);

			// Back face (z = -1.0f)
			Gl.Color3(1.0f, 1.0f, 0.0f);     // Yellow
			Gl.Vertex3(1.0f, -1.0f, -1.0f);
			Gl.Vertex3(-1.0f, -1.0f, -1.0f);
			Gl.Vertex3(-1.0f, 1.0f, -1.0f);
			Gl.Vertex3(1.0f, 1.0f, -1.0f);

			// Left face (x = -1.0f)
			Gl.Color3(0.0f, 0.0f, 1.0f);     // Blue
			Gl.Vertex3(-1.0f, 1.0f, 1.0f);
			Gl.Vertex3(-1.0f, 1.0f, -1.0f);
			Gl.Vertex3(-1.0f, -1.0f, -1.0f);
			Gl.Vertex3(-1.0f, -1.0f, 1.0f);

			// Right face (x = 1.0f)
			Gl.Color3(1.0f, 0.0f, 1.0f);     // Magenta
			Gl.Vertex3(1.0f, 1.0f, -1.0f);
			Gl.Vertex3(1.0f, 1.0f, 1.0f);
			Gl.Vertex3(1.0f, -1.0f, 1.0f);
			Gl.Vertex3(1.0f, -1.0f, -1.0f);
			Gl.End();  // End of drawing color-cube

			Gl.LoadIdentity();

			Gl.Translate(0.0f, 0.0f, -7.0f);
			Gl.Begin(PrimitiveType.Triangles);
			for (int i = 0; i < bunny.Length; i++)
			{
				ref var face = ref bunny[i];

				Gl.Normal3(face.Normal.X, face.Normal.Y, face.Normal.Z);
				Gl.Vertex3(face.A.X, face.A.Y, face.A.Z);
				Gl.Vertex3(face.B.X, face.B.Y, face.B.Z);
				Gl.Vertex3(face.C.X, face.C.Y, face.C.Z);
			}
			Gl.End();

			Gl.LoadIdentity();
		}

		private static void Initialize()
		{
			Gl.Light(LightName.Light0, LightParameter.Diffuse, new[] { 1.0f, 0.3f, 0.3f, 1.0f });
			Gl.Light(LightName.Light0, LightParameter.Position, new[] { 1.0f, 1.0f, -3.0f, 0.0f });

			Gl.Enable(EnableCap.Light0);
			Gl.Enable(EnableCap.Lighting);

			Gl.ClearColor(0.2f, 0.2f, 0.0f, 1.0f);
			Gl.ClearDepth(1.0f);
			Gl.Enable(EnableCap.DepthTest);
			Gl.DepthFunc(DepthFunction.Lequal);
			Gl.ShadeModel(ShadingModel.Smooth);
			Gl.FrontFace(FrontFaceDirection.Ccw);
			Gl.Hint(HintTarget.PerspectiveCorrectionHint, HintMode.Nicest);
		}
	}
}
