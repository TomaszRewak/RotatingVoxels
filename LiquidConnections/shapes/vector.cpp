#include "vector.h"

float Shapes::Vector::dotProduct(const Shapes::Vector& second) const
{
	return x * second.x + y * second.y + z * second.z;
}

Shapes::Vector Shapes::Vector::crossProduct(const Shapes::Vector& second) const
{
	return Vector(
		y * second.z - z * second.y,
		z * second.x - x * second.z,
		x * second.y - y * second.x
	);
}