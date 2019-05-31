#pragma once

#include "voxel-space.h"
#include "../shapes/shape.h"

namespace Voxel
{
	template<size_t X, size_t Y, size_t Z>
	class VoxelSpaceBuilder
	{
	public:
		VoxelSpace<X, Y, Z> voxelSpace;

		void add(const Shapes::Shape& shape)
		{
			std::shuffle(shape.faces.begin(), shape.faces.end());

			for (const auto& face : shape.faces)
				add(face);
		}

	private:
		void add(const Shapes::Face& face)
		{
			int minX = std::max(face.minX(), 0),
				minY = std::max(face.minY(), 0),
				minZ = std::max(face.minZ(), 0),
				maxX = std::min(face.maxX(), X),
				maxY = std::min(face.maxY(), Y),
				maxZ = std::min(face.maxZ(), Z);

			Shapes::Vertex intersection;

			for (int x = minX; x < maxX; x++)
				for (int y = minY, y < maxY; y++)
					if (face.intersect(Shapes::Ray(Shapes::Vertex(x, y, -1), Shapes::Vector(0, 0, 1)), intersection))
						addZ(x, y, face, intersection);

			for (int x = minX; x < maxX; x++)
				for (int z = minZ, z < maxZ; z++)
					if (face.intersect(Shapes::Ray(Shapes::Vertex(x, -1, z), Shapes::Vector(0, 1, 0)), intersection))
						addY(x, z, face, intersection);

			for (int y = minY; y < maxY; y++)
				for (int z = minZ, z < maxZ; z++)
					if (face.intersect(Shapes::Ray(Shapes::Vertex(-1, y, z), Shapes::Vector(1, 0, 0)), intersection))
						addX(y, z, face, intersection);
		}

		void getDistance(float d, float normal)
		{
			return normal < 0 ? d : -d;
		}

		void addX(int y, int z, const Shapes::Face& face, const Shapes::Vertex& intersection)
		{
			int x1 = (int)intersection.x,
				x2 = (int)intersection.x + 1;

			add(VoxelCoordinates(x1, y, z), getDistance(intersection.x - x1, face.normal.x), intersection);
			add(VoxelCoordinates(x2, y, z), getDistance(intersection.x - x2, face.normal.x), intersection);
		}

		void addY(int x, int z, const Shapes::Face& face, const Shapes::Vertex& intersection)
		{
			int y1 = (int)intersection.y,
				y2 = (int)intersection.y + 1;

			add(VoxelCoordinates(x, y1, z), getDistance(intersection.y - y1, face.normal.y), intersection);
			add(VoxelCoordinates(x, y2, z), getDistance(intersection.y - y2, face.normal.y), intersection);
		}

		void addZ(int x, int y, const Shapes::Face& face, const Shapes::Vertex& intersection)
		{
			int z1 = (int)intersection.z,
				z2 = (int)intersection.z + 1;

			add(VoxelCoordinates(x, y, z1), getDistance(intersection.z - z1, face.normal.z), intersection);
			add(VoxelCoordinates(x, y, z2), getDistance(intersection.z - z2, face.normal.z), intersection);
		}

		void add(const VoxelCoordinates& coordinates, float distance, const Shapes::Vertex& intersection)
		{
			if (voxelSpace[coordinates] < distance)
				return;

			voxelSpace[coordinates] = distance;

			propagate(coordinates, intersection);
		}

		void propagate(const VoxelCoordinates& coordinates, const Shapes::Vertex& intersection)
		{
			std::stack<VoxelCoordinates> points{
				coordinates.move(1, 0, 0),
				coordinates.move(-1, 0, 0),
				coordinates.move(0, 1, 0),
				coordinates.move(0, -1, 0),
				coordinates.move(0, 0, 1),
				coordinates.move(0, 0, -1)
			};

			while (!points.empty())
			{
				VoxelCoordinates point = points.top();
				points.pop();

				if (!inside(point))
					continue;

				float distance = std::sqrt(
					std::pow(intersection.x - point.x, 2) +
					std::pow(intersection.y - point.y, 2) +
					std::pow(intersection.z - point.z, 2));

				if (distance >= voxelSpace[point])
					continue;

				voxelSpace[point] = distance;

				points.push(point.move(1, 0, 0));
				points.push(point.move(-1, 0, 0));
				points.push(point.move(0, 1, 0));
				points.push(point.move(0, -1, 0));
				points.push(point.move(0, 0, 1));
				points.push(point.move(0, 0, -1));
			}
		}
	};
}