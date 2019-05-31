#include <algorithm>

#include "face.h"

float Shapes::Face::minX() const
{
	return std::min(vertices[0].x, std::min(vertices[1].x, vertices[2].x));
}

float Shapes::Face::minY() const
{
	return std::min(vertices[0].y, std::min(vertices[1].y, vertices[2].y));
}

float Shapes::Face::minZ() const
{
	return std::min(vertices[0].z, std::min(vertices[1].z, vertices[2].z));
}

float Shapes::Face::maxX() const
{
	return std::max(vertices[0].x, std::max(vertices[1].x, vertices[2].x));
}

float Shapes::Face::maxY() const
{
	return std::max(vertices[0].y, std::max(vertices[1].y, vertices[2].y));
}

float Shapes::Face::maxZ() const
{
	return std::max(vertices[0].z, std::max(vertices[1].z, vertices[2].z));
}

bool Shapes::Face::intersect(const Ray& ray, Vertex& intersection) const
{
	const float epsilon = 0.0000001;

	const Vector edge1 = Vector(vertices[0], vertices[1]);
	const Vector edge2 = Vector(vertices[0], vertices[2]);

	const Vector crossProduct = ray.vector.crossProduct(edge2);
	const float dotProduct = edge1.dotProduct(crossProduct);

	if (dotProduct > -epsilon && dotProduct < epsilon)
		return false;

	const float invertedDotProduct = 1.0 / dotProduct;
	const Vector vector0 = Vector(vertices[0], ray.origin);
	const float u = invertedDotProduct * vector0.dotProduct(crossProduct);

	if (u < 0.0 || u > 1.0)
		return false;

	const Vector q = vector0.crossProduct(edge1);
	const float v = invertedDotProduct * ray.vector.dotProduct(q);

	if (v < 0.0 || u + v > 1.0)
		return false;

	float distance = invertedDotProduct * edge2.dotProduct(q);
	if (distance > epsilon)
	{
		intersection = ray.intersect(distance);
		return true;
	}
	else
		return false;
}