using OpenGL;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Resources.Shaders
{
	class Shader
	{
		public uint GlShader { get; }

		public Shader(ShaderType type, string path)
		{
			GlShader = Gl.CreateShader(type);
			Gl.ShaderSource(GlShader, new[] { File.ReadAllText(path) });
			Gl.CompileShader(GlShader);

			Validate();
		}

		private void Validate()
		{
			Gl.GetShader(GlShader, ShaderParameterName.CompileStatus, out var success);

			if (success != 0)
				return;

			LogError();

			throw new InvalidProgramException();
		}

		private void LogError()
		{
			StringBuilder infoLog = new StringBuilder(1024);
			Gl.GetShaderInfoLog(GlShader, 1024, out int _, infoLog);
			Console.WriteLine("Errors: \n{0}", infoLog);
		}
	}
}
