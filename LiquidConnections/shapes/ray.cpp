#pragma once

#include "ray.h"

Shapes::Vertex Shapes::Ray::intersect(float distance) const
{
	return origin + vector * distance;
}