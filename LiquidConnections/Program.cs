﻿using Alea;
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
using System.IO;
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
			voxelSpaceBuilder.Add(ShapeNormalizer.NormalizeShape(bunny, new Bounds(0, 0, 0, 40, 40, 40)));

			voxelizedBunny = voxelSpaceBuilder.Build();

			var voxelSpaceConbiner = new VoxelSpaceCombiner(80, 40, 40);
			//voxelSpaceConbiner.Add(voxelizedBunny, new Vector(0, 0, 0));
			//voxelSpaceConbiner.Add(voxelizedBunny, new Vector(40, 0, 0));

			//voxelizedBunny = voxelSpaceConbiner.VoxelSpace;

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

		static uint colorDataBuffer;
		static float[] colorDataValues = {
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
		};

		static uint vertexDataBuffer;
		static float[] vertexDataValues = {
			-1.0f, -1.0f, +1.0f,
			+1.0f, -1.0f, +1.0f,
			+1.0f, +1.0f, +1.0f,
			-1.0f, +1.0f, +1.0f,
			-1.0f, -1.0f, -1.0f,
			+1.0f, -1.0f, -1.0f,
			+1.0f, +1.0f, -1.0f,
			-1.0f, +1.0f, -1.0f
		};

		static uint indexDataBuffer;
		static ushort[] indexDataValues = {
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

		static uint program;

		private static void Render(object sender, NativeWindowEventArgs e)
		{
			Gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

			Gl.MatrixMode(MatrixMode.Modelview);
			
			Gl.UseProgram(program);

			foreach (var coordinates in DiscreteBounds.Of(voxelizedBunny))
			{
				float distance = voxelizedBunny.At(coordinates).Distance;
				var normal = voxelizedBunny.At(coordinates).Normal.Normalize();
				float weight = distance < 1 ? 1f : distance < 2 ? 0.5f : 0f;
				//float weight = distance < 1 ? 0.5f : 0f;

				if (weight <= 0)
					continue;

				Gl.LoadIdentity();
				Gl.Translate(20 - coordinates.X, 20 - coordinates.Y, coordinates.Z - 80.0f);
				Gl.Scale(weight * 0.3f, weight * 0.3f, weight * 0.3f);
				Gl.MultMatrixf(LookAt(new Vertex3f(0, 0, 0), new Vertex3f(-normal.X, -normal.Y, -normal.Z), new Vertex3f(0, 1, 0)));
				Gl.Rotate(0, normal.X, normal.Y, normal.Z);

				Gl.EnableVertexAttribArray(0);
				Gl.BindBuffer(BufferTarget.ArrayBuffer, vertexDataBuffer);
				Gl.VertexAttribPointer(0, 3, VertexAttribType.Float, false, 0, IntPtr.Zero);
				//Gl.DrawArrays(PrimitiveType.Triangles, 0, 12 * 3);
				
				Gl.EnableVertexAttribArray(1);
				Gl.BindBuffer(BufferTarget.ArrayBuffer, colorDataBuffer);
				Gl.VertexAttribPointer(1, 3, VertexAttribType.Float, false, 0, IntPtr.Zero);

				Gl.BindBuffer(BufferTarget.ElementArrayBuffer, indexDataBuffer);

				Gl.DrawElementsInstanced(PrimitiveType.Triangles, 12 * 3, DrawElementsType.UnsignedShort, IntPtr.Zero, 1);
				Gl.DisableVertexAttribArray(0);
				Gl.DisableVertexAttribArray(1);

			}

			Gl.UseProgram(0);

			Gl.LoadIdentity();

			Gl.Translate(0.0f, 0.0f, -7.0f);

			Gl.LoadIdentity();
		}

		private static void Initialize()
		{
			Gl.Light(LightName.Light0, LightParameter.Diffuse, new[] { 0.7f, 0.7f, 0.7f, 1.0f });
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
			Gl.Enable(EnableCap.Normalize);

			vertexDataBuffer = Gl.GenBuffer();
			Gl.BindBuffer(BufferTarget.ArrayBuffer, vertexDataBuffer);
			Gl.BufferData(BufferTarget.ArrayBuffer, sizeof(float) * (uint)vertexDataValues.Length, vertexDataValues, BufferUsage.StaticDraw);

			colorDataBuffer = Gl.GenBuffer();
			Gl.BindBuffer(BufferTarget.ArrayBuffer, colorDataBuffer);
			Gl.BufferData(BufferTarget.ArrayBuffer, sizeof(float) * (uint)colorDataValues.Length, colorDataValues, BufferUsage.StaticDraw);

			indexDataBuffer = Gl.GenBuffer();
			Gl.BindBuffer(BufferTarget.ElementArrayBuffer, indexDataBuffer);
			Gl.BufferData(BufferTarget.ElementArrayBuffer, sizeof(ushort) * (uint)indexDataValues.Length, indexDataValues, BufferUsage.StaticDraw);

			var vertexShader = Gl.CreateShader(ShaderType.VertexShader);
			Gl.ShaderSource(vertexShader, new[] { File.ReadAllText("./Shaders/InstanceShader.vs") });
			Gl.CompileShader(vertexShader);

			Gl.GetShader(vertexShader, ShaderParameterName.CompileStatus, out var success1);
			if (success1 == 0)
			{
				StringBuilder infoLog = new StringBuilder(1024);
				Gl.GetShaderInfoLog(vertexShader, 1024, out int _, infoLog);
				Console.WriteLine("Errors: \n{0}", infoLog);
				throw new InvalidProgramException();
			}

			var fragmentShader = Gl.CreateShader(ShaderType.FragmentShader);
			Gl.ShaderSource(fragmentShader, new[] { File.ReadAllText("./Shaders/InstanceShader.fs") });
			Gl.CompileShader(fragmentShader);

			Gl.GetShader(fragmentShader, ShaderParameterName.CompileStatus, out var success2);
			if (success2 == 0)
			{
				StringBuilder infoLog = new StringBuilder(1024);
				Gl.GetShaderInfoLog(fragmentShader, 1024, out int _, infoLog);
				Console.WriteLine("Errors: \n{0}", infoLog);
				throw new InvalidProgramException();
			}

			program = Gl.CreateProgram();
			Gl.AttachShader(program, vertexShader);
			Gl.AttachShader(program, fragmentShader);
			Gl.LinkProgram(program);

			Gl.GetProgram(program, ProgramProperty.LinkStatus, out var success3);
			if (success3 == 0)
			{
				throw new InvalidProgramException();
			}
		}
	}
}
