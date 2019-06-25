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

			_program = Gl.CreateProgram();
			Gl.AttachShader(_program, vertexShader);
			Gl.AttachShader(_program, fragmentShader);
			Gl.LinkProgram(_program);

			Gl.GetProgram(_program, ProgramProperty.LinkStatus, out var success3);
			if (success3 == 0)
			{
				StringBuilder infoLog = new StringBuilder(1024);
				var error = Gl.GetError();
				Gl.GetProgramInfoLog(_program, 1024, out int _, infoLog);
				Console.WriteLine("Errors: \n{0}", infoLog);
				throw new InvalidProgramException();
			}
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

		struct ProgramContext : IDisposable
		{
			public void Dispose()
			{
				Gl.UseProgram(0);
			}
		}
	}
}
