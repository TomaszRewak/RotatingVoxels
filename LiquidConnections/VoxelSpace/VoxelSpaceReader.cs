using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
{
	static class VoxelSpaceReader
	{
		public static Face[] GenerateShape(VoxelCell[,,] voxelSpace)
		{
			var faces = new List<Face>();
			var bounds = new DiscreteBounds(voxelSpace).Offset(0, 0, 0, -1, -1, -1);

			foreach (var coordinates in bounds)
				GenerateFaces(voxelSpace, coordinates, faces);

			return faces.ToArray();
		}

		private static void GenerateFaces(VoxelCell[,,] voxelSpace, in DiscreteCoordinates coordinates, ICollection<Face> faces)
		{
			Span<DiscreteCoordinates> nearbyCoordinates = stackalloc DiscreteCoordinates[]
			{
				coordinates.Move(0, 0, 0),
				coordinates.Move(0, 0, 1),
				coordinates.Move(0, 1, 0),
				coordinates.Move(0, 1, 1),
				coordinates.Move(1, 0, 0),
				coordinates.Move(1, 0, 1),
				coordinates.Move(1, 1, 0),
				coordinates.Move(1, 1, 1),
			};

			Span<FaceVertex> vertices = stackalloc FaceVertex[8];
			int verticesCount = 0;

			foreach (var vertexCoordinates in nearbyCoordinates)
				if (HasNearbyVertex(voxelSpace, vertexCoordinates))
					vertices[verticesCount++] = GetNearbyVertex(voxelSpace, vertexCoordinates);

			for (int i = 0; i < verticesCount; i++)
				for (int j = i + 1; j < verticesCount; j++)
					for (int k = j + 1; k < verticesCount; k++)
						GenerateFace(vertices[i], vertices[j], vertices[k], faces);
		}

		private static bool HasNearbyVertex(VoxelCell[,,] voxelSpace, in DiscreteCoordinates coordinates)
		{
			return new Vector(coordinates.AsVertex(), voxelSpace.At(coordinates).NearestIntersection).Length < 1;
		}

		private static FaceVertex GetNearbyVertex(VoxelCell[,,] voxelSpace, in DiscreteCoordinates coordinates)
		{
			ref var cell = ref voxelSpace.At(coordinates);

			return new FaceVertex(cell.NearestIntersection, cell.Normal * 100);
		}

		private static void GenerateFace(
			in FaceVertex vertexA,
			in FaceVertex vertexB,
			in FaceVertex vertexC, 
			ICollection<Face> faces)
		{
			if (!ValidateFaceCandidate(vertexA, vertexB, vertexC))
				return;

			faces.Add(new Face(vertexA, vertexB, vertexC));
		}

		private static bool ValidateFaceCandidate(in FaceVertex vertexA, in FaceVertex vertexB, in FaceVertex vertexC)
		{
			return
				vertexA.Point != vertexB.Point &&
				vertexB.Point != vertexC.Point &&
				vertexA.Normal.DotProduct(vertexB.Normal) >= 0 &&
				vertexA.Normal.DotProduct(vertexC.Normal) >= 0 &&
				vertexB.Normal.DotProduct(vertexC.Normal) >= 0;
		}
	}
}
