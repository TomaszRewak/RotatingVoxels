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
		private readonly FullVoxelCell[,,] _voxelSpace;

		public VoxelSpaceBuilder(int x, int y, int z)
		{
			_voxelSpace = new FullVoxelCell[x, y, z];
			
			Clear();
		}

		public void Clear()
		{
			foreach (var coordinates in DiscreteBounds.Of(_voxelSpace))
				_voxelSpace.At(coordinates) = new FullVoxelCell(Vertex.Max, new Vector());
		}

		public VoxelCell[,,] Build()
		{
			var voxels = new VoxelCell[_voxelSpace.GetLength(0), _voxelSpace.GetLength(1), _voxelSpace.GetLength(2)];

			foreach (var coordinates in DiscreteBounds.Of(_voxelSpace))
				voxels.At(coordinates) = Build(coordinates);

			return voxels;
		}

		private VoxelCell Build(in DiscreteCoordinates coordinates)
		{
			ref var source = ref _voxelSpace.At(coordinates);

			return new VoxelCell
			{
				Distance = new Vector(coordinates.AsVertex(), source.NearestIntersection).Length,
				Normal = source.Normal
			};
		}

		public void Add(Face[] faces)
		{
			for (int i = 0; i < faces.Length; i++)
				Add(faces[i]);

			Propagate();
		}

		private void Add(in Face face)
		{
			var bounds = DiscreteBounds
				.Of(_voxelSpace)
				.Clip(new DiscreteBounds(new Bounds(face)));

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
			if (!DiscreteBounds.Of(_voxelSpace).Inside(coordinates))
				return;

			var oldDistance = new Vector(coordinates.AsVertex(), _voxelSpace.At(coordinates).NearestIntersection).Length;
			var newDistance = new Vector(coordinates.AsVertex(), intersection).Length;

			if (newDistance >= oldDistance)
				return;

			ref var cell = ref _voxelSpace.At(coordinates);

			cell = new FullVoxelCell(intersection, normal);
		}

		private void Propagate()
		{
			var bounds = DiscreteBounds.Of(_voxelSpace);

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int y = bounds.MinY; y <= bounds.MaxY; y++)
					for (int z = bounds.MinZ; z < bounds.MaxZ; z++)
						Propagate(new DiscreteCoordinates(x, y, z), new DiscreteCoordinates(x, y, z + 1));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int y = bounds.MinY; y <= bounds.MaxY; y++)
					for (int z = bounds.MaxZ; z > bounds.MinZ; z--)
						Propagate(new DiscreteCoordinates(x, y, z), new DiscreteCoordinates(x, y, z - 1));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					for (int y = bounds.MinY; y < bounds.MaxY; y++)
						Propagate(new DiscreteCoordinates(x, y, z), new DiscreteCoordinates(x, y + 1, z));

			for (int x = bounds.MinX; x <= bounds.MaxX; x++)
				for (int z = bounds.MinZ; z <= bounds.MaxZ; z++)
					for (int y = bounds.MaxY; y > bounds.MinY; y--)
						Propagate(new DiscreteCoordinates(x, y, z), new DiscreteCoordinates(x, y - 1, z));

			for (int y = bounds.MinY; y <= bounds.MaxY; y++)
				for (int z = bounds.MinZ; z < bounds.MaxZ; z++)
					for (int x = bounds.MinX; x < bounds.MaxX; x++)
						Propagate(new DiscreteCoordinates(x, y, z), new DiscreteCoordinates(x + 1, y, z));

			for (int y = bounds.MinY; y <= bounds.MaxY; y++)
				for (int z = bounds.MinZ; z < bounds.MaxZ; z++)
					for (int x = bounds.MaxX; x > bounds.MinX; x--)
						Propagate(new DiscreteCoordinates(x, y, z), new DiscreteCoordinates(x - 1, y, z));
		}

		private void Propagate(in DiscreteCoordinates from, in DiscreteCoordinates to)
		{
			ref var sourceCell = ref _voxelSpace.At(from);
			ref var destinationCell = ref _voxelSpace.At(to);

			var oldDistance = new Vector(to.AsVertex(), destinationCell.NearestIntersection).Length;
			var newDistance = new Vector(to.AsVertex(), sourceCell.NearestIntersection).Length;

			if (newDistance >= oldDistance)
				return;

			destinationCell = sourceCell;
		}
	}
}
