using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
{
	class VoxelSpaceBuilder
	{
		public VoxelCell[,,] VoxelSpace { get; }

		public DiscreteBounds Bounds => new DiscreteBounds(VoxelSpace);

		public VoxelSpaceBuilder(int x, int y, int z)
		{
			VoxelSpace = new VoxelCell[x, y, z];

			Clear();
		}

		public void Clear()
		{
			foreach (var coordinates in Bounds)
				VoxelSpace.At(coordinates) = new VoxelCell(Vertex.Max, new Vector());
		}

		public void Add(Face[] faces)
		{
			for (int i = 0; i < faces.Length; i++)
				Add(faces[i]);
		}

		private void Add(in Face face)
		{
			var bounds = Bounds.Clip(new DiscreteBounds(new Bounds(face)));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int y = bounds.MinY; y <= bounds.MaxY; y++)
					if (new Ray(new Vertex(x, y, -1), new Vector(0, 0, 1)).Intersect(face, out var intersection))
						Add(intersection, new DiscreteCoordinates(x, y, (int)intersection.Z), face.Normal(intersection));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					if (new Ray(new Vertex(x, -1, z), new Vector(0, 1, 0)).Intersect(face, out var intersection))
						Add(intersection, new DiscreteCoordinates(x, (int)intersection.Y, z), face.Normal(intersection));

			for (int y = bounds.MinY; y <= bounds.MaxY; y++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					if (new Ray(new Vertex(-1, y, z), new Vector(1, 0, 0)).Intersect(face, out var intersection))
						Add(intersection, new DiscreteCoordinates((int)intersection.X, y, z), face.Normal(intersection));
		}

		private void Add(in Vertex intersection, in DiscreteCoordinates coordinates, in Vector normal)
		{
			if (!Bounds.Inside(coordinates))
				return;

			var oldDistance = new Vector(coordinates.AsVertex(), VoxelSpace.At(coordinates).NearestIntersection).Length;
			var newDistance = new Vector(coordinates.AsVertex(), intersection).Length;

			if (newDistance >= oldDistance)
				return;

			ref var cell = ref VoxelSpace.At(coordinates);

			cell = new VoxelCell(intersection, (normal + cell.Normal * cell.Weight) / (cell.Weight + 1), cell.Weight + 1);
		}

		//private void propagate(const DiscreteCoordinates& coordinates, const Shapes::Vertex& intersection)
		//{
		//	std::stack<DiscreteCoordinates> points;
		//	points.push(coordinates.move(1, 0, 0));
		//	points.push(coordinates.move(-1, 0, 0));
		//	points.push(coordinates.move(0, 1, 0));
		//	points.push(coordinates.move(0, -1, 0));
		//	points.push(coordinates.move(0, 0, 1));
		//	points.push(coordinates.move(0, 0, -1));

		//	while (!points.empty())
		//	{
		//		DiscreteCoordinates point = points.top();
		//		points.pop();

		//		if (!voxelSpace.inside(point))
		//			continue;

		//		float distance = std::sqrt(
		//			std::pow(intersection.x - point.x, 2) +
		//			std::pow(intersection.y - point.y, 2) +
		//			std::pow(intersection.z - point.z, 2));

		//		if (distance >= voxelSpace[point])
		//			continue;

		//		voxelSpace[point] = distance;

		//		points.push(point.move(1, 0, 0));
		//		points.push(point.move(-1, 0, 0));
		//		points.push(point.move(0, 1, 0));
		//		points.push(point.move(0, -1, 0));
		//		points.push(point.move(0, 0, 1));
		//		points.push(point.move(0, 0, -1));
		//	}
		//}
	}
}
