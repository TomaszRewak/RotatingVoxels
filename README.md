# RotatingVoxels

In this project I've used C# combined with Alea GPU and OpenGL.Net to create a simple, hardware-accelerated, 3d animation of rotating cubes.

<p align="center">
  <img src="https://github.com/TomaszRewak/RotatingVoxels/blob/master/About/rotating-voxels.gif?raw=true" width=400/>
</p>

The premise here is simple. I take a 3d object, voxelize it (describe it in terms of a homogenous grid) and transform each voxel (each grid cell) into a visual cube.

On a more technical level:
- the voxelization is a preprocessing stage,
- generated voxel space is transformed each frame using an updated transformation matrix (this step is done in CUDA)
- OpenGL and its instantiating mechanism takes care of displaying all of the cubes in the real time.
