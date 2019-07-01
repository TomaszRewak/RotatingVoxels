using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RotatingVoxels.Window
{
	class FpsCounter
	{
		private readonly Stopwatch _stopwatch = new Stopwatch();

		private int _fps;

		public void Start()
		{
			_stopwatch.Start();
		}

		public void Tick()
		{
			_fps++;

			if (_stopwatch.ElapsedMilliseconds >= 1000)
			{
				Log();

				_stopwatch.Restart();
				_fps = 0;
			}
		}

		private void Log()
		{
			Console.WriteLine(_fps * 1000f / _stopwatch.ElapsedMilliseconds);
		}
	}
}
