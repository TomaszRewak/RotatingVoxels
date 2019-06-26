using OpenGL;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Resources.Shaders
{
	class ShadingProgram
	{
		private readonly uint _program;

		public ShadingProgram()
		{
			_program = Gl.CreateProgram();

			var vertexShader = new Shader(ShaderType.VertexShader, "./Shaders/InstanceShader.vs");
			var fragmentShader = new Shader(ShaderType.FragmentShader, "./Shaders/InstanceShader.fs");

			Gl.AttachShader(_program, vertexShader.GlShader);
			Gl.AttachShader(_program, fragmentShader.GlShader);
			Gl.LinkProgram(_program);

			Validate();
		}

		public IDisposable Use()
		{
			Gl.UseProgram(_program);

			return new ProgramContext();
		}

		public Matrix4x4f Transformation
		{
			set => Gl.UniformMatrix4f(Gl.GetUniformLocation(_program, "transformation"), 1, false, value);
		}

		public uint Weights
		{
			set => Gl.Uniform1i(Gl.GetUniformLocation(_program, "weights"), 1, value);
		}

		private void Validate()
		{
			Gl.GetProgram(_program, ProgramProperty.LinkStatus, out var success);

			if (success != 0)
				return;

			LogError();
			
			throw new InvalidProgramException();
		}

		private void LogError()
		{
			StringBuilder infoLog = new StringBuilder(1024);
			var error = Gl.GetError();
			Gl.GetProgramInfoLog(_program, 1024, out int _, infoLog);
			Console.WriteLine("Errors: \n{0}", infoLog);
		}

		struct ProgramContext : IDisposable
		{
			public void Dispose()
			{
				Gl.UseProgram(0);
			}
		}
	}
}
