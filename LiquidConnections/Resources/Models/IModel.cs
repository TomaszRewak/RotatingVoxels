using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Resources.Models
{
	interface IModel : IDisposable
	{
		void Draw(int times);
	}
}
