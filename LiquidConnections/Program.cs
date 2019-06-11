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
using System.Runtime.InteropServices;
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

		static VoxelCell[,,] voxelizedBunny;
		static Face[] bunny;
		static float[] bunnyFloat;

		static void Main(string[] args)
		{
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

			Stopwatch stopwatch = Stopwatch.StartNew();

			var voxelSpaceBuilder = new VoxelSpaceBuilder(40, 40, 40);
			voxelSpaceBuilder.Add(ShapeNormalizer.NormalizeShape(bunny, new Bounds(10, 10, 10, 30, 30, 30)));

			voxelizedBunny = voxelSpaceBuilder.Build();

			var voxelSpaceConbiner = new VoxelSpaceCombiner(80, 40, 40);
			//voxelSpaceConbiner.Add(voxelizedBunny, new Vector(0, 0, 0));
			//voxelSpaceConbiner.Add(voxelizedBunny, new Vector(40, 0, 0));

			bunny = ShapeNormalizer.NormalizeShape(VoxelSpaceReader.GenerateShape(voxelSpaceConbiner.VoxelSpace), new Bounds(-2, -2, -2, 2, 2, 2));

			bunnyFloat = MemoryMarshal.Cast<Face, float>(bunny).ToArray();

			stopwatch.Stop();
			Console.WriteLine($"Pricessing took: {stopwatch.ElapsedMilliseconds}ms");

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

		static Matrix4x4f LookAt(in Vertex3f eye, in Vertex3f target, in Vertex3f upDir)
		{
			// compute the forward vector from target to eye
			Vertex3f forward = eye - target;
			forward.Normalize();                 // make unit length

			// compute the left vector
			Vertex3f left = upDir ^ forward; // cross product
			left.Normalize();

			// recompute the orthonormal up vector
			Vertex3f up = forward ^ left;    // cross product

			// init 4x4 matrix
			Matrix4x4f matrix = Matrix4x4f.Identity;

			// set rotation part, inverse rotation matrix: M^-1 = M^T for Euclidean transform
			matrix[0, 0] = left.x;
			matrix[1, 0] = left.y;
			matrix[2, 0] = left.z;
			matrix[0, 1] = up.x;
			matrix[1, 1] = up.y;
			matrix[2, 1] = up.z;
			matrix[0, 2] = forward.x;
			matrix[1, 2] = forward.y;
			matrix[2, 2] = forward.z;

			// set translation part
			matrix[3, 0] = -left.x * eye.x - left.y * eye.y - left.z * eye.z;
			matrix[3, 1] = -up.x * eye.x - up.y * eye.y - up.z * eye.z;
			matrix[3, 2] = -forward.x * eye.x - forward.y * eye.y - forward.z * eye.z;

			return matrix;
		}

		static int offset = 0;
		private static void Render(object sender, NativeWindowEventArgs e)
		{
			Gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

			Gl.MatrixMode(MatrixMode.Modelview);

			Gl.LoadIdentity();
			Gl.Translate(Math.Cos(offset * 0.01) * 3, Math.Cos(offset * 0.023) * 3, -7.0f);
			offset++;

			//Gl.Begin(PrimitiveType.Quads);

			//Gl.Color3(0.0f, 1.0f, 0.0f);     // Green
			//Gl.Vertex3(1.0f, 1.0f, -1.0f);
			//Gl.Vertex3(-1.0f, 1.0f, -1.0f);
			//Gl.Vertex3(-1.0f, 1.0f, 1.0f);
			//Gl.Vertex3(1.0f, 1.0f, 1.0f);

			//// Bottom face (y = -1.0f)
			//Gl.Color3(1.0f, 0.5f, 0.0f);     // Orange
			//Gl.Vertex3(1.0f, -1.0f, 1.0f);
			//Gl.Vertex3(-1.0f, -1.0f, 1.0f);
			//Gl.Vertex3(-1.0f, -1.0f, -1.0f);
			//Gl.Vertex3(1.0f, -1.0f, -1.0f);

			//// Front face  (z = 1.0f)
			//Gl.Color3(1.0f, 0.0f, 0.0f);     // Red
			//Gl.Vertex3(1.0f, 1.0f, 1.0f);
			//Gl.Vertex3(-1.0f, 1.0f, 1.0f);
			//Gl.Vertex3(-1.0f, -1.0f, 1.0f);
			//Gl.Vertex3(1.0f, -1.0f, 1.0f);

			//// Back face (z = -1.0f)
			//Gl.Color3(1.0f, 1.0f, 0.0f);     // Yellow
			//Gl.Vertex3(1.0f, -1.0f, -1.0f);
			//Gl.Vertex3(-1.0f, -1.0f, -1.0f);
			//Gl.Vertex3(-1.0f, 1.0f, -1.0f);
			//Gl.Vertex3(1.0f, 1.0f, -1.0f);

			//// Left face (x = -1.0f)
			//Gl.Color3(0.0f, 0.0f, 1.0f);     // Blue
			//Gl.Vertex3(-1.0f, 1.0f, 1.0f);
			//Gl.Vertex3(-1.0f, 1.0f, -1.0f);
			//Gl.Vertex3(-1.0f, -1.0f, -1.0f);
			//Gl.Vertex3(-1.0f, -1.0f, 1.0f);

			//// Right face (x = 1.0f)
			//Gl.Color3(1.0f, 0.0f, 1.0f);     // Magenta
			//Gl.Vertex3(1.0f, 1.0f, -1.0f);
			//Gl.Vertex3(1.0f, 1.0f, 1.0f);
			//Gl.Vertex3(1.0f, -1.0f, 1.0f);
			//Gl.Vertex3(1.0f, -1.0f, -1.0f);
			//Gl.End();  // End of drawing color-cube

			Gl.LoadIdentity();
			Gl.Translate(0, 0, -7.0f);

			foreach (var coordinates in DiscreteBounds.Of(voxelizedBunny))
			{
				float distance = voxelizedBunny.At(coordinates).Distance;
				var normal = voxelizedBunny.At(coordinates).Normal.Normalize();
				float weight = distance < 1 ? 0.6f : distance < 2 ? 0.3f : distance < 3 ? 0.1f : 0f;
				//float weight = distance < 1 ? 0.5f : 0f;

				if (weight <= 0)
					continue;

				Gl.LoadIdentity();
				Gl.Translate(20 - coordinates.X, 20 - coordinates.Y, coordinates.Z - 50.0f);
				Gl.Scale(weight, weight, weight);
				Gl.MultMatrixf(LookAt(new Vertex3f(0, 0, 0), new Vertex3f(-normal.X, -normal.Y, -normal.Z), new Vertex3f(0, 1, 0)));
				Gl.Rotate(0, normal.X, normal.Y, normal.Z);

				Gl.Begin(PrimitiveType.Quads);
				// top
				Gl.Color3(0.3f, 0.3f, 0.3f);
				Gl.Normal3(0.0f, 1.0f, 0.0f);
				Gl.Vertex3(-0.5f, 0.5f, 0.5f);
				Gl.Vertex3(0.5f, 0.5f, 0.5f);
				Gl.Vertex3(0.5f, 0.5f, -0.5f);
				Gl.Vertex3(-0.5f, 0.5f, -0.5f);

				Gl.End();

				Gl.Begin(PrimitiveType.Quads);
				// front
				Gl.Color3(0.3f, 1.0f, 0.3f);
				Gl.Normal3(0.0f, 0.0f, 1.0f);
				Gl.Vertex3(0.5f, -0.5f, 0.5f);
				Gl.Vertex3(0.5f, 0.5f, 0.5f);
				Gl.Vertex3(-0.5f, 0.5f, 0.5f);
				Gl.Vertex3(-0.5f, -0.5f, 0.5f);

				Gl.End();

				Gl.Begin(PrimitiveType.Quads);
				// right
				Gl.Color3(0.3f, 0.3f, 0.3f);
				Gl.Normal3(1.0f, 0.0f, 0.0f);
				Gl.Vertex3(0.5f, 0.5f, -0.5f);
				Gl.Vertex3(0.5f, 0.5f, 0.5f);
				Gl.Vertex3(0.5f, -0.5f, 0.5f);
				Gl.Vertex3(0.5f, -0.5f, -0.5f);

				Gl.End();

				Gl.Begin(PrimitiveType.Quads);
				// left
				Gl.Color3(0.3f, 0.3f, 0.3f);
				Gl.Normal3(-1.0f, 0.0f, 0.0f);
				Gl.Vertex3(-0.5f, -0.5f, 0.5f);
				Gl.Vertex3(-0.5f, 0.5f, 0.5f);
				Gl.Vertex3(-0.5f, 0.5f, -0.5f);
				Gl.Vertex3(-0.5f, -0.5f, -0.5f);

				Gl.End();

				Gl.Begin(PrimitiveType.Quads);
				// bottom
				Gl.Color3(0.3f, 0.3f, 0.3f);
				Gl.Normal3(0.0f, -1.0f, 0.0f);
				Gl.Vertex3(0.5f, -0.5f, 0.5f);
				Gl.Vertex3(-0.5f, -0.5f, 0.5f);
				Gl.Vertex3(-0.5f, -0.5f, -0.5f);
				Gl.Vertex3(0.5f, -0.5f, -0.5f);

				Gl.End();

				Gl.Begin(PrimitiveType.Quads);
				// back
				Gl.Color3(0.3f, 0.3f, 1.0f);
				Gl.Normal3(0.0f, 0.0f, -1.0f);
				Gl.Vertex3(0.5f, 0.5f, -0.5f);
				Gl.Vertex3(0.5f, -0.5f, -0.5f);
				Gl.Vertex3(-0.5f, -0.5f, -0.5f);
				Gl.Vertex3(-0.5f, 0.5f, -0.5f);

				Gl.End();
			}

			Gl.LoadIdentity();

			Gl.Translate(0.0f, 0.0f, -7.0f);
			Gl.Begin(PrimitiveType.Triangles);

			var bunnySpan = bunny.AsSpan();
			for (int i = 0; i < bunnySpan.Length; i++)
			{
				ref var face = ref bunnySpan[i];

				Gl.Normal3(Math.Sin(offset * 0.003) * face.A.Normal.Z + Math.Cos(offset * 0.003) * face.A.Normal.X, face.A.Normal.Y, -Math.Sin(offset * 0.003) * face.A.Normal.X + Math.Cos(offset * 0.003) * face.A.Normal.Z);
				Gl.Vertex3(Math.Sin(offset * 0.003) * face.A.Point.Z + Math.Cos(offset * 0.003) * face.A.Point.X, face.A.Point.Y, -Math.Sin(offset * 0.003) * face.A.Point.X + Math.Cos(offset * 0.003) * face.A.Point.Z);
				Gl.Normal3(Math.Sin(offset * 0.003) * face.B.Normal.Z + Math.Cos(offset * 0.003) * face.B.Normal.X, face.B.Normal.Y, -Math.Sin(offset * 0.003) * face.B.Normal.X + Math.Cos(offset * 0.003) * face.B.Normal.Z);
				Gl.Vertex3(Math.Sin(offset * 0.003) * face.B.Point.Z + Math.Cos(offset * 0.003) * face.B.Point.X, face.B.Point.Y, -Math.Sin(offset * 0.003) * face.B.Point.X + Math.Cos(offset * 0.003) * face.B.Point.Z);
				Gl.Normal3(Math.Sin(offset * 0.003) * face.C.Normal.Z + Math.Cos(offset * 0.003) * face.C.Normal.X, face.C.Normal.Y, -Math.Sin(offset * 0.003) * face.C.Normal.X + Math.Cos(offset * 0.003) * face.C.Normal.Z);
				Gl.Vertex3(Math.Sin(offset * 0.003) * face.C.Point.Z + Math.Cos(offset * 0.003) * face.C.Point.X, face.C.Point.Y, -Math.Sin(offset * 0.003) * face.C.Point.X + Math.Cos(offset * 0.003) * face.C.Point.Z);
			}
			Gl.End();

			Gl.LoadIdentity();
		}

		private static void Initialize()
		{
			Gl.Light(LightName.Light0, LightParameter.Diffuse, new[] { 1.0f, 0.3f, 0.3f, 1.0f });
			Gl.Light(LightName.Light0, LightParameter.Position, new[] { 1.0f, 1.0f, 13.0f, 0.0f });
			Gl.Enable(EnableCap.ColorMaterial);

			Gl.Enable(EnableCap.Light0);
			Gl.Enable(EnableCap.Lighting);

			Gl.ClearColor(0.2f, 0.2f, 0.0f, 1.0f);
			Gl.ClearDepth(1.0f);
			Gl.Enable(EnableCap.DepthTest);
			Gl.DepthFunc(DepthFunction.Lequal);
			Gl.ShadeModel(ShadingModel.Smooth);
			Gl.Hint(HintTarget.PerspectiveCorrectionHint, HintMode.Nicest);
		}
	}
}
