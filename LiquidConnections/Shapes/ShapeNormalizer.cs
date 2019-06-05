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
					A = minPoint + new Vector(faces[i].A, maxShapePoint) * scaleVector,
					B = minPoint + new Vector(faces[i].B, maxShapePoint) * scaleVector,
					C = minPoint + new Vector(faces[i].C, maxShapePoint) * scaleVector,
					Normal = faces[i].Normal / scaleVector
				};

			return normalizedFaces;
		}
	}
}
