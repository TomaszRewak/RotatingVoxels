#include <regex>
#include <string>

#include "stl-loader.h"

namespace LiquidConnections {
	namespace Stl {
		Shapes::Shape ShapeLoader::load(std::experimental::filesystem::path path)
		{
			std::ifstream file(path);
			std::string fileContent(
				(std::istreambuf_iterator<char>(file)),
				(std::istreambuf_iterator<char>()));

			return loadShape(fileContent);
		}

		Shapes::Shape ShapeLoader::loadShape(std::string & file)
		{
			std::regex pattern(R"(facet\s+normal\s+(\S+)\s+(\S+)\s+(\S+)\s+outer loop\s+vertex\s+(\S+)\s+(\S+)\s+(\S+)\s+vertex\s+(\S+)\s+(\S+)\s+(\S+)\s+vertex\s+(\S+)\s+(\S+)\s+(\S+)\s+endloop\s+endfacet)");
			std::smatch match;

			Shapes::Shape shape;

			auto fileBegin = file.cbegin();
			while (std::regex_search(fileBegin, file.cend(), match, pattern))
			{
				shape.faces.push_back(Shapes::Face{
					Shapes::Vertex(std::stof(match[4]), std::stof(match[5]), std::stof(match[6])),
					Shapes::Vertex(std::stof(match[7]), std::stof(match[8]), std::stof(match[9])),
					Shapes::Vertex(std::stof(match[10]), std::stof(match[11]), std::stof(match[12])),
					Shapes::Vector(std::stof(match[1]), std::stof(match[2]), std::stof(match[3]))
				});
			}

			return shape;
		}
	}
}