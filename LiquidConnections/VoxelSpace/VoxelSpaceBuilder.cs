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
		public static VoxelCell[,,] Build(Face[] faces, DiscreteBounds bounds)
		{
			var voxelSpace = new VoxelCell[bounds.Width, bounds.Height, bounds.Depth];

			Clear(voxelSpace);
			Add(voxelSpace, faces);
			Propagate(voxelSpace);

			return voxelSpace;
		}

		private static void Add(VoxelCell[,,] voxelSpac, Face[] faces)
		{
			for (int i = 0; i < faces.Length; i++)
				Add(voxelSpac, faces[i]);
		}

		private static void Clear(VoxelCell[,,] voxelSpace)
		{
			foreach (var coordinates in DiscreteBounds.Of(voxelSpace))
				voxelSpace.At(coordinates) = new VoxelCell(Vertex.MaxValue, new Vector());
		}

		private static void Add(VoxelCell[,,] voxelSpace, in Face face)
		{
			var bounds = DiscreteBounds
				.Of(voxelSpace)
				.Clip(new DiscreteBounds(new Bounds(face)));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int y = bounds.MinY; y <= bounds.MaxY; y++)
					if (new Ray(Vertex.At(x, y, -1), Vector.To(0, 0, 1)).Intersect(face, out var intersection))
						Add(voxelSpace, intersection, DiscreteCoordinates.At(x, y, (int)intersection.Z), face.Normal(intersection));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					if (new Ray(Vertex.At(x, -1, z), Vector.To(0, 1, 0)).Intersect(face, out var intersection))
						Add(voxelSpace, intersection, DiscreteCoordinates.At(x, (int)intersection.Y, z), face.Normal(intersection));

			for (int y = bounds.MinY; y <= bounds.MaxY; y++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					if (new Ray(Vertex.At(-1, y, z), Vector.To(1, 0, 0)).Intersect(face, out var intersection))
						Add(voxelSpace, intersection, DiscreteCoordinates.At((int)intersection.X, y, z), face.Normal(intersection));
		}

		private static void Add(VoxelCell[,,] voxelSpace, in Vertex intersection, in DiscreteCoordinates coordinates, in Vector normal)
		{
			if (!DiscreteBounds.Of(voxelSpace).Inside(coordinates))
				return;

			var oldDistance = Vector.Between(coordinates.AsVertex(), voxelSpace.At(coordinates).NearestIntersection).Length;
			var newDistance = Vector.Between(coordinates.AsVertex(), intersection).Length;

			if (newDistance >= oldDistance)
				return;

			ref var cell = ref voxelSpace.At(coordinates);

			cell = new VoxelCell(intersection, normal);
		}

		private static void Propagate(VoxelCell[,,] voxelSpace)
		{
			var bounds = DiscreteBounds.Of(voxelSpace);

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int y = bounds.MinY; y <= bounds.MaxY; y++)
					for (int z = bounds.MinZ; z < bounds.MaxZ; z++)
						Propagate(voxelSpace, DiscreteCoordinates.At(x, y, z), DiscreteCoordinates.At(x, y, z + 1));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int y = bounds.MinY; y <= bounds.MaxY; y++)
					for (int z = bounds.MaxZ; z > bounds.MinZ; z--)
						Propagate(voxelSpace, DiscreteCoordinates.At(x, y, z), DiscreteCoordinates.At(x, y, z - 1));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					for (int y = bounds.MinY; y < bounds.MaxY; y++)
						Propagate(voxelSpace, DiscreteCoordinates.At(x, y, z), DiscreteCoordinates.At(x, y + 1, z));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					for (int y = bounds.MaxY; y > bounds.MinY; y--)
						Propagate(voxelSpace, DiscreteCoordinates.At(x, y, z), DiscreteCoordinates.At(x, y - 1, z));

			for (int y = bounds.MinY; y <= bounds.MaxY; y++)
				for (int z = bounds.MinZ; z < bounds.MaxZ; z++)
					for (int x = bounds.MinX; x < bounds.MaxX; x++)
						Propagate(voxelSpace, DiscreteCoordinates.At(x, y, z), DiscreteCoordinates.At(x + 1, y, z));

			for (int y = bounds.MinY; y <= bounds.MaxY; y++)
				for (int z = bounds.MinZ; z < bounds.MaxZ; z++)
					for (int x = bounds.MaxX; x > bounds.MinX; x--)
						Propagate(voxelSpace, DiscreteCoordinates.At(x, y, z), DiscreteCoordinates.At(x - 1, y, z));
		}

		private static void Propagate(VoxelCell[,,] voxelSpace, in DiscreteCoordinates from, in DiscreteCoordinates to)
		{
			ref var sourceCell = ref voxelSpace.At(from);
			ref var destinationCell = ref voxelSpace.At(to);

			var oldDistance = Vector.Between(to.AsVertex(), destinationCell.NearestIntersection).Length;
			var newDistance = Vector.Between(to.AsVertex(), sourceCell.NearestIntersection).Length;

			if (newDistance >= oldDistance)
				return;

			destinationCell = sourceCell;
		}
	}
}
