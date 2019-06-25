using Alea;
using Alea.CSharp;
using LiquidConnections.Geometry;
using LiquidConnections.Models;
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

		private static void BunnyKernel(deviceptr<VoxelFace> weightsBauffer, VoxelCell[] shape, float xOffset)
		{
			var length = 40 * 40 * 40;
			var start = length * (threadIdx.x) / blockDim.x;
			var end = length * (threadIdx.x + 1) / blockDim.x;
			int offset = (int)Math.Floor(xOffset);

			//weightsBauffer.Set(0, 0.5f);

			//for (int i = 0; i < gpuBunny.Length; i++)
			//	texture.Set(i, bunnyFloat[i]);

			for (int i = start; i < end; i++)
			{
				int x = i % 40 + offset;
				int y = i / 40 % 40;
				int z = i / 40 / 40 % 40;

				var coordinates1 = DiscreteCoordinates.At( x      % 40, y, z);
				var coordinates2 = DiscreteCoordinates.At((x + 1) % 40, y, z);
				var bounds = DiscreteBounds.OfSize(40, 40, 40);

				xOffset = xOffset - (float)Math.Floor(xOffset);

				ref var cell1 = ref shape[bounds.Index(coordinates1)];
				var distance1 = Vector.Between(coordinates1.AsVertex(), cell1.NearestIntersection).Length;

				ref var cell2 = ref shape[bounds.Index(coordinates2)];
				var distance2 = Vector.Between(coordinates2.AsVertex(), cell2.NearestIntersection).Length;

				var weight1 = Math.Max(0f, 1f - distance1 );
				var weight2 = Math.Max(0f, 1f - distance2 );

				weightsBauffer.Set(i, new VoxelFace
				{
					Weight = weight1 * (1 - xOffset) + weight2 * xOffset,
					Normal = cell1.Normal * (1 - xOffset) + cell2.Normal * xOffset
				});
			}

		}

		private static void CopyToTexture(IntPtr ptr, float xOffset)
		{
			var lp = new LaunchParam(1, 256);
			Gpu.Default.Launch(BunnyKernel, lp, new deviceptr<VoxelFace>(ptr), gpuBunny, xOffset * 0.2f);
		}

		private static void Clear(deviceptr<VoxelFace> weightsBauffer)
		{
			var length = 40 * 40 * 40;
			var start = length * (threadIdx.x) / blockDim.x;
			var end = length * (threadIdx.x + 1) / blockDim.x;

			for (int i = start; i < end; i++)
				weightsBauffer.Set(i, new VoxelFace { Weight = 0 });

		}

		private static void Clear(IntPtr ptr)
		{
			var lp = new LaunchParam(1, 256);
			Gpu.Default.Launch(Clear, lp, new deviceptr<VoxelFace>(ptr));
		}

		static VoxelCell[,,] voxelizedBunny;
		static Face[] bunny;
		static float[] bunnyFloat;
		static VoxelCell[] gpuBunny;
		static VoxelCell[] gpuCombined;

		static void Main(string[] args)
		{
			bunny = StlReader.LoadShape("./Examples/ball.stl");

			Stopwatch stopwatch = Stopwatch.StartNew();

			voxelizedBunny = VoxelSpaceBuilder.Build(ShapeNormalizer.NormalizeShape(bunny, new Bounds(15, 15, 15, 25, 25, 25)), DiscreteBounds.OfSize(40, 40, 40));

			bunnyFloat = MemoryMarshal.Cast<Face, float>(bunny).ToArray();

			stopwatch.Stop();
			Console.WriteLine($"Pricessing took: {stopwatch.ElapsedMilliseconds}ms");

			var bounds = DiscreteBounds.Of(voxelizedBunny);
			var flatBunny = new VoxelCell[bounds.Length];
			foreach (var coordinates in bounds)
				flatBunny[bounds.Index(coordinates)] = voxelizedBunny.At(coordinates);

			gpuBunny = Gpu.Default.Allocate<VoxelCell>(40 * 40 * 40);
			Gpu.Copy(flatBunny, gpuBunny);

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

		static Box box;

		static uint program;
		static uint weightsBuffer;
		static uint weightsTexture;

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

			Gl.UseProgram(program);

			unsafe
			{
				IntPtr a;
				IntPtr b;
				CUDAInterop.cuGLRegisterBufferObject(weightsBuffer);
				CUDAInterop.cuSafeCall(CUDAInterop.cuGLMapBufferObject(&a, &b, weightsBuffer));
				Clear(a);
				CopyToTexture(a, iteration * 0.1f);
				Gpu.Default.Synchronize();
				CUDAInterop.cuGLUnmapBufferObject(weightsBuffer);
				CUDAInterop.cuGLUnregisterBufferObject(weightsBuffer);
			}

			{
				float n = 1f;
				float f = 10000f;
				float r = 0.3f;
				float t = 0.3f;

				Matrix4x4f transformation = new Matrix4x4f(
					n / r, 0, 0, 0,
					0, n / t, 0, 0,
					0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
					0, 0, -1, 0
					);

				transformation = transformation * Matrix4x4f.LookAt(new Vertex3f((float)Math.Sin(iteration * 0.000001) * 1f, -(float)Math.Sin(iteration * 0.003) * 0.6f, (float)Math.Cos(iteration * 0.000001) * 1f), new Vertex3f(0, 0, 0), new Vertex3f(0, -1, 0));

				Gl.UniformMatrix4f(Gl.GetUniformLocation(program, "transformation"), 1, false, transformation);

				Gl.ActiveTexture(TextureUnit.Texture1);
				Gl.BindTexture((TextureTarget)Gl.TEXTURE_BUFFER, weightsTexture);
				Gl.TexBuffer((TextureTarget)Gl.TEXTURE_BUFFER, InternalFormat.Rgba32f, weightsBuffer);
				Gl.Uniform1i(Gl.GetUniformLocation(program, "weights"), 1, weightsTexture);

				using (var boxContext = box.Bind())
				{
					Gl.DrawElementsInstanced(PrimitiveType.Triangles, 12 * 3, DrawElementsType.UnsignedShort, IntPtr.Zero, 40 * 40 * 40);
				}

			}

			Gl.UseProgram(0);

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

			Gl.Enable(EnableCap.Multisample);
			Gl.Enable(EnableCap.Blend);

			Gl.ClearColor(0.2f, 0.2f, 0.2f, 1.0f);
			Gl.ClearDepth(1.0f);
			Gl.Enable(EnableCap.DepthTest);
			Gl.DepthFunc(DepthFunction.Lequal);
			Gl.ShadeModel(ShadingModel.Smooth);
			Gl.Hint(HintTarget.PerspectiveCorrectionHint, HintMode.Nicest);
			Gl.Enable(EnableCap.Normalize);

			box = new Box();

			weightsBuffer = Gl.GenBuffer();
			weightsTexture = Gl.GenTexture();

			Gl.BindBuffer(BufferTarget.TextureBuffer, weightsBuffer);
			Gl.BufferData(BufferTarget.TextureBuffer, (uint)Marshal.SizeOf<VoxelFace>() * 40 * 40 * 40, null, BufferUsage.DynamicDraw);

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
				StringBuilder infoLog = new StringBuilder(1024);
				var error = Gl.GetError();
				Gl.GetProgramInfoLog(program, 1024, out int _, infoLog);
				Console.WriteLine("Errors: \n{0}", infoLog);
				throw new InvalidProgramException();
			}

			stopwatch.Start();
		}
	}
}
