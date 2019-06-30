using RotatingVoxels.Geometry;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Stl
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
				var normal = Vector.To(values[0], values[1], values[2]).Normalize();

				faces[i] = new Face
				{
					A = new FaceVertex(Vertex.At(values[3], values[4], values[5]), normal),
					B = new FaceVertex(Vertex.At(values[6], values[7], values[8]), normal),
					C = new FaceVertex(Vertex.At(values[9], values[10], values[11]), normal)
				};
			}

			return faces;
		}
	}
}
