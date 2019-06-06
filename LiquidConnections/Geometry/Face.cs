﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Geometry
{
	struct Face
	{
		public FaceVertex A;
		public FaceVertex B;
		public FaceVertex C;

		public Face(FaceVertex a, FaceVertex b, FaceVertex c)
		{
			A = a;
			B = b;
			C = c;
		}
	}
}
