using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Shapes
{
	static class ShapeNormalizer
	{
		public static Face[] NormalizeShape(Face[] faces, in Bounds bounds)
		{
			var normalizedFaces = new Face[faces.Length];
			var shapeBounds = Bounds.Min;

			foreach (var face in faces)
				shapeBounds += new Bounds(face);

			var minPoint = new Vertex(bounds.MinX, bounds.MinY, bounds.MinZ);
			var maxPoint = new Vertex(bounds.MaxX, bounds.MaxY, bounds.MaxZ);
			var minShapePoint = new Vertex(shapeBounds.MinX, shapeBounds.MinY, shapeBounds.MinZ);
			var maxShapePoint = new Vertex(shapeBounds.MaxX, shapeBounds.MaxY, shapeBounds.MaxZ);

			var scaleVector = new Vector(minPoint, maxPoint) / new Vector(minShapePoint, maxShapePoint);

			for (int i = 0; i < faces.Length; i++)
				normalizedFaces[i] = new Face
				{
					A = Normalize(faces[i].A, minPoint, maxShapePoint, scaleVector),
					B = Normalize(faces[i].B, minPoint, maxShapePoint, scaleVector),
					C = Normalize(faces[i].C, minPoint, maxShapePoint, scaleVector),
				};

			return normalizedFaces;
		}

		private static FaceVertex Normalize(in FaceVertex vertex, in Vertex minPoint, in Vertex maxShapePoint, in Vector scaleVector)
		{
			return new FaceVertex
			{
				Point = minPoint + new Vector(vertex.Point, maxShapePoint) * scaleVector,
				Normal = vertex.Normal / scaleVector
			};
		}
	}
}
