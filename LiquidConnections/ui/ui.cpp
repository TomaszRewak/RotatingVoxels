#include "ui.h"

#include <helper_gl.h>

#include <GL/freeglut.h>
#include <GL/glew.h>

void LiquidConnections::UI::init(int argc, char **argv)
{
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA);
	glutCreateWindow(argv[0]);

	//glutDisplayFunc(displayFunc);
	//glutReshapeFunc(reshapeFunc);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glewInit();
}