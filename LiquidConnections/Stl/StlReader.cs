using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Stl
{
	class StlReader
	{
		public static Face[] LoadShape(string path)
		{
			using (FileStream fileStream = File.OpenRead(path))
			using (BinaryReader reader = new BinaryReader(fileStream))
			{
				reader.BaseStream.Seek(80, SeekOrigin.Begin);

				return LoadShape(reader);
			}
		}

		private static Face[] LoadShape(BinaryReader reader)
		{
			var faces = new Face[reader.ReadInt32()];

			for (int i = 0; i < faces.Length; i++)
			{
				var bytes = reader.ReadBytes(50);
				var values = MemoryMarshal.Cast<byte, float>(bytes);

				faces[i] = new Face
				{
					A = new Vertex(values[3], values[4], values[5]),
					B = new Vertex(values[6], values[7], values[8]),
					C = new Vertex(values[9], values[10], values[11]),
					Normal = new Vector(values[0], values[1], values[2])
				};
			}

			return faces;
		}
	}
}
