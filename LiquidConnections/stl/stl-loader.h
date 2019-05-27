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
			static void loadHeader(std::ifstream& file);
			static Shapes::Shape loadShape(std::ifstream& file);
			static Shapes::Face loadFace(std::ifstream& file);
			static Shapes::Vertex loadVertex(std::ifstream& file);
		};
	}
}