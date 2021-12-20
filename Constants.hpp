#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#pragma once

#include "structs.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//screen properties
const int SCREEN_FPS = 30;
#define screenWidth 480
#define screenHeight 480
#define viewWidth 19.0f
#define viewHeight (screenHeight*viewWidth/screenWidth)

/*
#define yield 0.08f
#define stiffness 0.18f
#define nearStiffness 0.01f
#define linearViscocity 0.5f
#define quadraticViscocity 1.f


//Particle size
#define particleRadius 0.05f
#define particleHeight (6*particleRadius)

//Particle rendering properties
#define drawRatio 3.0f
#define velocityFactor 0.001f
*/

// simulation parameters in constant memory


#define maxParticles 59050//6000

//Particle size
#define particleRadius 0.035f
#define particleHeight (6*particleRadius)

//Particle rendering properties
#define drawRatio 3.0f
#define velocityFactor 1.001f

#define frameRate 37
#define timeStep 3
#define Pi 3.14159265f
#define deltaTime ((1.0f/frameRate) / timeStep)

#define restDensity 75.0f
#define surfaceTension 0.0006f
#define multipleParticleTypes true
/*
SDL_Window* screen = NULL;
SDL_GLContext gContext;

//world boundaries
Boundary boundaries[4] =
{
    Boundary(1, 0, 0),
    Boundary(0, 1, 0),
    Boundary(-1, 0, -viewWidth),
    Boundary(0, -1, -viewHeight)
};
*/
//types of materials, rgb orders lightest to heaviest, and the sources that will use them
#define redParticle ParticleType(Vec4(0.5f, 0.71f, 0.1f, 1), 1.75f)
#define greenParticle ParticleType(Vec4(0.1f, 0.75f, 0.1f, 1), 1.2f)
#define blueParticle ParticleType(Vec4(0.1f, 0.1f, 0.75f, 1), 1.4f)


//the map of the scene
const int mapWidth = (int)(viewWidth / particleHeight);
const int mapHeight = (int)(viewHeight / particleHeight);




#endif // CONSTANTS_HPP
