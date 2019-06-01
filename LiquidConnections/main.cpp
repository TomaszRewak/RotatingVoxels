#include "ui/ui.h"
#include "stl/stl-loader.h"
#include "voxel/voxel-shape-preprocesor.h"
#include "voxel/voxel-space-builder.h"

void copyShape()
{

}

int main(int argc, char **argv)
{
	const size_t X = 100, Y = 100, Z = 100;

	auto model = Stl::ShapeLoader::load("../examples/bunny.stl");
	auto scaledModel = Voxel::VoxelShapePreprocesor::normalizeShape<X, Y, Z>(model);
	
	auto voxelSpaceBuilder = std::make_shared<Voxel::VoxelSpaceBuilder<X, Y, Z>>();
	voxelSpaceBuilder->add(scaledModel);

	UI::init(argc, argv);
	//findCudaGLDevice(argc, (const char**)argv);

	//createTexture();
	//createPixelBuffer();
	//createDataBuffer();

	//glutMainLoop();
	//timerEvent(0);
}