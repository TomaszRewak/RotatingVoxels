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
		public static Face[] GenerateShape(float[,,] voxelSpace)
		{
			var faces = new List<Face>();
			var bounds = new DiscreteBounds(voxelSpace).Offset(0, 0, 0, -1, -1, -1);

			foreach (var coordinates in bounds)
				GenerateFaces(voxelSpace, coordinates, faces);

			return faces.ToArray();
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
				new DiscreteEdge(coordinates.Move(0, 1, 0), coordinates.Move(1, 1, 0)),
				new DiscreteEdge(coordinates.Move(0, 0, 1), coordinates.Move(1, 0, 1)),
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

			// TODO: Simplify normal vector calculation
			var vectorA = new Vector(crossingA.Origin, crossingB.Origin);
			var vectorB = new Vector(crossingA.Origin, crossingC.Origin);
			var normalVector = vectorA.CrossProduct(vectorB);
			var directionVector = crossingA.Vector + crossingB.Vector + crossingC.Vector;

			if (normalVector.DotProduct(directionVector) < 0)
				normalVector = -normalVector;

			faces.Add(new Face {
				A = crossingA.Origin,
				B = crossingB.Origin,
				C = crossingC.Origin,
				Normal = normalVector
			});
		}

		private static Ray Crossing(float[,,] voxelSpace, in DiscreteEdge edge)
		{
			var normalizedEdge = NormalizeEdge(voxelSpace, edge);

			var d1 = voxelSpace.At(normalizedEdge.Begin);
			var d2 = voxelSpace.At(normalizedEdge.End);

			var point = normalizedEdge.Begin.AsVertex() - normalizedEdge.AsVector() * d1 / (d2 - d1);

			return new Ray(point, new Vector(normalizedEdge.Begin.AsVertex(), point));
		}

		private static DiscreteEdge NormalizeEdge(float[,,] voxelSpace, in DiscreteEdge edge)
		{
			if (voxelSpace.At(edge.Begin) > voxelSpace.At(edge.End))
				return new DiscreteEdge(edge.End, edge.Begin);
			else
				return edge;
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
