#include <regex>
#include <string>

#include "stl-loader.h"

namespace LiquidConnections {
	namespace Stl {
		Shapes::Shape ShapeLoader::load(std::experimental::filesystem::path path)
		{
			std::ifstream file(path, std::ios::in | std::ios::binary);
			file.seekg(80);

			return loadShape(file);
		}

		Shapes::Shape ShapeLoader::loadShape(std::ifstream& file)
		{
			Shapes::Shape shape;

			int faces;
			file.read((char*)&faces, 4);

			for (int i = 0; i < faces; i++)
			{
				float xn, yn, zn;
				float x1, y1, z1;
				float x2, y2, z2;
				float x3, y3, z3;
				int _;

				file.read((char*)&xn, 4);
				file.read((char*)&yn, 4);
				file.read((char*)&zn, 4);

				file.read((char*)&x1, 4);
				file.read((char*)&y1, 4);
				file.read((char*)&z1, 4);

				file.read((char*)&x2, 4);
				file.read((char*)&y2, 4);
				file.read((char*)&z2, 4);

				file.read((char*)&x3, 4);
				file.read((char*)&y3, 4);
				file.read((char*)&z3, 4);

				file.read((char*)&_, 2);

				shape.faces.push_back(Shapes::Face{
					Shapes::Vertex(x1, y1, z1),
					Shapes::Vertex(x2, y2, z2),
					Shapes::Vertex(x3, y3, z3),
					Shapes::Vector(xn, yn, zn)
				});
			}

			return shape;
		}
	}
}