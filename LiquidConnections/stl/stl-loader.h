#pragma once

#include <filesystem>
#include <fstream>

#include "../shapes/shape.h"

namespace Stl
{
	class ShapeLoader
	{
	public:
		static Shapes::Shape load(std::experimental::filesystem::path path);

	private:
		static Shapes::Shape loadShape(std::ifstream& file);
	};
}