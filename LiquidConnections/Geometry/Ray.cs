using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Geometry
{
	struct Ray
	{
		public Vertex Origin;
		public Vector Vector;

		public Ray(Vertex origin, Vector vector)
		{
			Origin = origin;
			Vector = vector;
		}

		public Vertex At(float distance)
		{
			return Origin + Vector * distance;
		}

		public bool Intersect(in Face face, out Vertex intersection)
		{
			intersection = new Vertex();

			var epsilon = 0.0000001f;

			var edge1 = new Vector(face.A.Point, face.B.Point);
			var edge2 = new Vector(face.A.Point, face.C.Point);

			var crossProduct = Vector.CrossProduct(edge2);
			var dotProduct = edge1.DotProduct(crossProduct);

			if (dotProduct > -epsilon && dotProduct < epsilon)
				return false;

			var invertedDotProduct = 1.0f / dotProduct;
			var rayVector = new Vector(face.A.Point, Origin);
			var u = invertedDotProduct * rayVector.DotProduct(crossProduct);

			if (u < 0.0 || u > 1.0)
				return false;

			var q = rayVector.CrossProduct(edge1);
			var v = invertedDotProduct * Vector.DotProduct(q);

			if (v < 0.0 || u + v > 1.0)
				return false;

			float distance = invertedDotProduct * edge2.DotProduct(q);
			if (distance > epsilon)
			{
				intersection = At(distance);
				return true;
			}
			else
				return false;
		}
	}
}
