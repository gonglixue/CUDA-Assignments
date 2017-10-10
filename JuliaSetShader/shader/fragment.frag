#version 330 core
in vec3 ourColor;
in vec2 TexCoord; // 0-1000

out vec4 color;

int DIM = 1000;

void main()
{
	int x = int(TexCoord.x);
	int y = int(TexCoord.y);
    
    const float Mag2Limit = 1.9f;
    const float scale = 1.5;
    float jx = scale * (DIM/2.0 - x) / (DIM/2.0);
    float jy = scale * (DIM/2.0 - y) / (DIM/2.0);

    float c0=-0.8, ci=0.156;
    float a0=jx, ai=jy;

    int flag = 1;
    float mag;
    for(int i=0; i<200; i++)
    {
    	float new_a0, new_ai;
    	new_a0 = a0*a0 - ai*ai;
    	new_ai = ai*a0 + a0*ai;

    	a0 = new_a0 + c0;
    	ai = new_ai + ci;

    	mag = a0*a0 + ai*ai;
    	if(mag > Mag2Limit) 
    	{
    		flag = 0;
    		break;
    	}
    }

    color = vec4(
    	1.0 * flag * (Mag2Limit/(mag+0.8)),
    	1.0 * flag * (mag/Mag2Limit),
    	1.0 * flag * (mag + 0.8)/Mag2Limit ,
    	1
    );
    //color = vec4(x/1000.0, y/1000.0, 0, 1);

}