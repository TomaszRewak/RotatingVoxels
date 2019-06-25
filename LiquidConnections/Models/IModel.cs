using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LiquidConnections.Models
{
	interface IModel : IDisposable
	{
		IDisposable Bind();
	}
}
