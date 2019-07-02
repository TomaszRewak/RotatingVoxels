#version 330 core

out vec4 FragColor;
  
in vec3 fColor;
in vec3 fPos;
in float fWeight;

void main()
{
    float d = 0;

    float ax = abs(fPos.x);
    float ay = abs(fPos.y);
    float az = abs(fPos.z);

    if (ax > 0.99)
        d = max(ay, az);
    else if (ay > 0.99)
        d = max(ax, az);
    else
        d = max(ax, ay);

    d = min(1, max(0, d + 0.25 / fWeight - 1) * fWeight * 4);
        
    gl_FragColor = vec4(fColor, 1.0) * (1 - d);  
        
}