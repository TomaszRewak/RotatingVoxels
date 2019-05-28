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

			file.read((char*)&shape.facesCount, 4);
			
			shape.faces = new Shapes::Face[shape.facesCount];

			for (int i = 0; i < shape.facesCount; i++)
			{
				auto& face = shape.faces[i];

				file.read((char*)&face.normal.x, 4);
				file.read((char*)&face.normal.y, 4);
				file.read((char*)&face.normal.z, 4);

				file.read((char*)&face.vertices[0].x, 4);
				file.read((char*)&face.vertices[0].y, 4);
				file.read((char*)&face.vertices[0].z, 4);

				file.read((char*)&face.vertices[1].x, 4);
				file.read((char*)&face.vertices[1].y, 4);
				file.read((char*)&face.vertices[1].z, 4);

				file.read((char*)&face.vertices[2].x, 4);
				file.read((char*)&face.vertices[2].y, 4);
				file.read((char*)&face.vertices[2].z, 4);

				int _;
				file.read((char*)&_, 2);
			}

			return shape;
		}
	}
}