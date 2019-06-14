using LiquidConnections.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.VoxelSpace
{
	class VoxelSpaceCombiner
	{
		const float _distance = 10;

		public VoxelCell[,,] VoxelSpace { get; }

		public VoxelSpaceCombiner(int x, int y, int z)
		{
			VoxelSpace = new VoxelCell[x, y, z];

			Clear();
		}

		public void Clear()
		{
			foreach (var coordinates in DiscreteBounds.Of(VoxelSpace))
				VoxelSpace.At(coordinates) = new VoxelCell(_distance, new Vector());
		}

		public void Add(VoxelCell[,,] shape, in Vector offset)
		{
			foreach (var coordinates in DiscreteBounds.Of(VoxelSpace))
				Add(shape, coordinates, offset);
		}

		public void Add(VoxelCell[,,] shape, in DiscreteCoordinates coordinates, in Vector offset)
		{
			ref var target = ref VoxelSpace.At(coordinates);
			var probe = Probe(shape, coordinates.AsVertex() - offset);

			if (probe.Distance > _distance)
				return;

			target = new VoxelCell
			{
				Normal = (target.Normal + probe.Normal).Normalize(),
				Distance = Math.Min(target.Distance, probe.Distance)
			};
		}

		private VoxelCell Probe(VoxelCell[,,] shape, in Vertex point)
		{
			VoxelCell result = new VoxelCell(float.MaxValue, new Vector());

			var bounds = new DiscreteBounds(DiscreteCoordinates.Floor(point))
			 .Offset(0, 0, 0, 1, 1, 1)
			 .Clip(DiscreteBounds.Of(shape));

			var normal = new Vector();
			var distance = 0f;
			var totalWeight = 0f;

			foreach (var coordinates in bounds)
			{
				return shape.At(coordinates);

				var weight = Math.Abs(1 - Math.Abs((point.X - coordinates.X) * (point.Y - coordinates.Y) * (point.Z - coordinates.Z)));

				normal += shape.At(coordinates).Normal * weight;
				distance += shape.At(coordinates).Distance * weight;

				totalWeight += weight;
			}

			return new VoxelCell
			{
				Distance = _distance,
				Normal = new Vector(1, 0, 0)
			};
		}
	}
}
