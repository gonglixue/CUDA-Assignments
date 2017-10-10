#version 330 core
in vec3 ourColor;
in vec2 TexCoord;

out vec4 color;

uniform sampler2D ourTexture1;
uniform float width;
uniform float height;
uniform int choice;  // 0:x  1:y 2:xandy

vec4 sobel(vec2 texcoord)
{
	int x = int(texcoord.x * width);
	int y = int(texcoord.y * height);
	vec4 result = vec4(0);
	
	mat3 sobel;
	if(choice == 0)
		sobel = mat3(-1, -2, -1, 0,0,0, 1, 2, 1);
	else if(choice == 1)
		sobel = mat3(1,0,-1, 2,0,-2, 1,0,-1);
		//sobel = mat3(0,0,0,0,1,0,0,0,0);
	for(int i=-1; i<=1; i++)  //x
	{
		for(int j=-1; j<=1; j++)  //y
		{
			vec2 tc = vec2( (x+i)/width, (y+j)/height );
			vec4 c = texture2D(ourTexture1, tc);
			float sobel_factor = sobel[j+1][i+1];
			result = sobel_factor * c + result;
		}
	}
	
	return result;
}

void main()
{
    //color = texture2D(ourTexture1, TexCoord);
	//color = vec4(ourColor,1.0);
	color = sobel(TexCoord);
}