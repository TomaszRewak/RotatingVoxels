#include <regex>
#include <string>

#include "stl-loader.h"

namespace LiquidConnections {
	namespace Stl {
		Shapes::Shape ShapeLoader::load(std::experimental::filesystem::path path)
		{
			std::ifstream file(path);

			loadHeader(file);

			return loadShape(file);
		}

		void ShapeLoader::loadHeader(std::ifstream & file)
		{
			std::string _;
			std::getline(file, _);
		}

		Shapes::Shape ShapeLoader::loadShape(std::ifstream & file)
		{
			Shapes::Shape shape;

			while (file.eof)
			{
				auto face = loadFace(file);
				shape.faces.push_back(face);
			}

			return shape;
		}

		Shapes::Face ShapeLoader::loadFace(std::ifstream & file)
		{
			// facet\s+normal\s+(\S+)\s+(\S+)\s+(\S+)([\s\S]*?)endfacet

			std::string _;

			file >> _;
			if (_ != "outer") throw std::exception();
			file >> _;
			if (_ != "loop") throw std::exception();

			Shapes::Vertex
				a = loadVertex(file),
				b = loadVertex(file),
				c = loadVertex(file);

			file >> _;
			if (_ != "endloop") throw std::exception();

			return Shapes::Face(a, b, c);
		}

		Shapes::Vertex ShapeLoader::loadVertex(std::ifstream & file)
		{
			std::string _;
			float x, y, z;

			file >> _ >> x >> y >> z;
			if (_ != "vertex") throw std::exception();

			return Shapes::Vertex(x, y, z);
		}
	}
}