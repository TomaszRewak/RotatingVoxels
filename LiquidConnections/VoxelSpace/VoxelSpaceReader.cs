using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
{
	static class VoxelSpaceReader
	{
		public static List<Face> GenerateShape(float[,,] voxelSpace)
		{
			var faces = new List<Face>();
			var bounds = new DiscreteBounds(voxelSpace).Offset(0, 0, 0, -1, -1, -1);

			foreach (var coordinates in bounds)
				GenerateFaces(voxelSpace, coordinates, faces);

			return faces;
		}

		private static void GenerateFaces(float[,,] voxelSpace, in DiscreteCoordinates coordinates, ICollection<Face> faces)
		{
			Span<DiscreteEdge> edges = stackalloc DiscreteEdge[]
			{
				new DiscreteEdge(coordinates.Move(0, 0, 0), coordinates.Move(0, 0, 1)),
				new DiscreteEdge(coordinates.Move(0, 0, 0), coordinates.Move(0, 1, 0)),
				new DiscreteEdge(coordinates.Move(0, 0, 0), coordinates.Move(1, 0, 0)),

				new DiscreteEdge(coordinates.Move(1, 1, 1), coordinates.Move(1, 1, 0)),
				new DiscreteEdge(coordinates.Move(1, 1, 1), coordinates.Move(1, 0, 1)),
				new DiscreteEdge(coordinates.Move(1, 1, 1), coordinates.Move(0, 1, 1)),

				new DiscreteEdge(coordinates.Move(1, 0, 0), coordinates.Move(1, 1, 0)),
				new DiscreteEdge(coordinates.Move(1, 0, 0), coordinates.Move(1, 0, 1)),

				new DiscreteEdge(coordinates.Move(0, 1, 1), coordinates.Move(0, 1, 0)),
				new DiscreteEdge(coordinates.Move(0, 1, 1), coordinates.Move(0, 0, 1)),
			};

			FilterEdges(voxelSpace, ref edges);

			for (int i = 0; i < edges.Length; i++)
				for (int j = i + 1; j < edges.Length; j++)
					for (int k = j + 1; k < edges.Length; k++)
						GenerateFace(voxelSpace, edges[i], edges[j], edges[k], faces);
		}

		private static void FilterEdges(float[,,] voxelSpace, ref Span<DiscreteEdge> edges)
		{
			for (int i = 0; i < edges.Length; i++)
			{
				if (CrossesZero(voxelSpace, edges[i]))
					continue;

				edges[i--] = edges[edges.Length - 1];
				edges = edges.Slice(0, edges.Length - 1);
			}
		}

		private static bool CrossesZero(float[,,] voxelSpace, in DiscreteEdge edge)
		{
			return voxelSpace.At(edge.Begin) * voxelSpace.At(edge.End) <= 0;
		}

		private static void GenerateFace(
			float[,,] voxelSpace, 
			in DiscreteEdge edgeA,
			in DiscreteEdge edgeB,
			in DiscreteEdge edgeC, 
			ICollection<Face> faces)
		{
			var crossingA = Crossing(voxelSpace, edgeA);
			var crossingB = Crossing(voxelSpace, edgeB);
			var crossingC = Crossing(voxelSpace, edgeC);

			if (!ValidateFaceCandidate(crossingA.Vector, crossingB.Vector, crossingC.Vector))
				return;

			faces.Add(new Face {
				A = crossingA.Origin,
				B = crossingB.Origin,
				C = crossingC.Origin,
				Normal = crossingA.Vector + crossingB.Vector + crossingC.Vector
			});
		}

		private static Ray Crossing(float[,,] voxelSpace, in DiscreteEdge edge)
		{
			var d1 = voxelSpace.At(edge.Begin);
			var d2 = voxelSpace.At(edge.End);

			var point = edge.Begin.AsVertex() + edge.AsVector() * d1 / (d1 + d2);

			if (d1 < d2)
				return new Ray(point, new Vector(edge.Begin.AsVertex(), point));
			else
				return new Ray(point, new Vector(edge.End.AsVertex(), point));
		}

		private static bool ValidateFaceCandidate(in Vector vectorA, in Vector vectorB, in Vector vectorC)
		{
			return
				vectorA.DotProduct(vectorB) >= 0 &&
				vectorA.DotProduct(vectorC) >= 0 &&
				vectorB.DotProduct(vectorC) >= 0;
		}
	}
}
