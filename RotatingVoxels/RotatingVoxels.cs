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
		static FpsCounter _fpsCounter = new FpsCounter();
		static Stopwatch _timer = new Stopwatch();
		static NativeWindow _window;

		static IScene _scene;

		static void Main(string[] args)
		{
			//_scene = new BunnyScene();
			//_scene = new BallScene();
			_scene = new RotatingScene();
			_scene.Load();

			using (_window = NativeWindow.Create())
			{
				InitializeWindow();
				InitializeOpenGl();

				_scene.Initialize();

				_fpsCounter.Start();
				_timer.Start();

				_window.Show();
				_window.Run();
			}
		}

		private static void Render(object sender, NativeWindowEventArgs e)
		{
			_fpsCounter.Tick();

			Gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

			_scene.Draw(_window.Width, _window.Height, _timer.Elapsed);
		}

		private static void InitializeWindow()
		{
			_window.DepthBits = 24;
			_window.Create(100, 100, 1200, 800, NativeWindowStyle.None);
			_window.Render += Render;
		}

		private static void InitializeOpenGl()
		{
			Gl.ClearColor(0.05f, 0.1f, 0.15f, 1.0f);
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
