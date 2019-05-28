#pragma once

#include <filesystem>
#include <fstream>

#include "../shepes/shape.h"

namespace LiquidConnections
{
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
}