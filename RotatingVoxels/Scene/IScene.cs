using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Scene
{
	interface IScene
	{
		void Load();
		void Initialize();
		void Draw(float width, float height, TimeSpan time);
	}
}
