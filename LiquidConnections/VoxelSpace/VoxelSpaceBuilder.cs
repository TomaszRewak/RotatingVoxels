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
				VoxelSpace.At(coordinates) = new VoxelCell(float.MaxValue, new Vector());
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
						AddZ(x, y, face, intersection, face.Normal(intersection));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					if (new Ray(new Vertex(x, -1, z), new Vector(0, 1, 0)).Intersect(face, out var intersection))
						AddY(x, z, face, intersection, face.Normal(intersection));

			for (int y = bounds.MinY; y <= bounds.MaxY; y++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					if (new Ray(new Vertex(-1, y, z), new Vector(1, 0, 0)).Intersect(face, out var intersection))
						AddX(y, z, face, intersection, face.Normal(intersection));
		}

		private float GetDistance(float d, float normal)
		{
			return normal < 0 ? d : -d;
		}

		private void AddX(int y, int z, in Face face, in Vertex intersection, in Vector normal)
		{
			int x1 = (int)intersection.X,
				x2 = (int)intersection.X + 1;

			Add(new DiscreteCoordinates(x1, y, z), GetDistance(intersection.X - x1, normal.X), normal);
			Add(new DiscreteCoordinates(x2, y, z), GetDistance(intersection.X - x2, normal.X), normal);
		}

		private void AddY(int x, int z, in Face face, in Vertex intersection, in Vector normal)
		{
			int y1 = (int)intersection.Y,
				y2 = (int)intersection.Y + 1;

			Add(new DiscreteCoordinates(x, y1, z), GetDistance(intersection.Y - y1, normal.Y), normal);
			Add(new DiscreteCoordinates(x, y2, z), GetDistance(intersection.Y - y2, normal.Y), normal);
		}

		private void AddZ(int x, int y, in Face face, in Vertex intersection, in Vector normal)
		{
			int z1 = (int)intersection.Z,
				z2 = (int)intersection.Z + 1;

			Add(new DiscreteCoordinates(x, y, z1), GetDistance(intersection.Z - z1, normal.Z), normal);
			Add(new DiscreteCoordinates(x, y, z2), GetDistance(intersection.Z - z2, normal.Z), normal);
		}

		private void Add(in DiscreteCoordinates coordinates, float distance, in Vector normal)
		{
			if (!Bounds.Inside(coordinates))
				return;

			if (VoxelSpace.At(coordinates).Distance < distance)
				return;

			VoxelSpace.At(coordinates) = new VoxelCell(distance, normal);
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
