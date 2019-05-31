#include "vector.h"

float Shapes::Vector::dotProduct(const Vector& second) const
{
	return x * second.x + y * second.y + z * second.z;
}

Shapes::Vector Shapes::Vector::crossProduct(const Vector& second) const
{
	return Vector(
		y * second.z - z * second.y,
		z * second.x - x * second.z,
		x * second.y - y * second.x
	);
}

Shapes::Vector Shapes::Vector::operator*(float by) const
{
	return Vector(x * by, y * by, z * by);
}

Shapes::Vertex operator+(const Shapes::Vertex& vertex, const Shapes::Vector& vector)
{
	return Shapes::Vertex(
		vertex.x + vector.x,
		vertex.y + vector.y,
		vertex.z + vector.z
	);
}