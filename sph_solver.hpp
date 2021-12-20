#ifndef SPH_SOLVER_HPP
#define SPH_SOLVER_HPP


#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "array"

#include "Constants.hpp"

class SPH_solver
{
public:
    SPH_solver();

    // general Particle properties
    float  yield=0.08f;
    float  stiffness= 0.18f;
    float  nearStiffness= 0.01f;
    float  linearViscocity =0.5f;
    float  quadraticViscocity= 1.f;

    float temp_increase=0.05f;
    float temp_decrease=0.005f;
    float gravity_coeff_max=11.0f;
    float gravity_coeff_min=3.0f;

    int maxMapCoeff1=mapWidth*mapHeight;
    int maxMapCoeff2=mapWidth*mapHeight;



    Source sources[3] =
    {
        Source(redParticle, Vec2(0.05f*viewWidth, 0.8f*viewHeight), Vec2(4, 1), 1.5),
        Source(blueParticle, Vec2(viewWidth - 0.05f*viewWidth, 0.8f*viewHeight), Vec2(-4, 1), 1.5),
        Source(greenParticle, Vec2(0, 0), Vec2(0, -1), 1.5),
    };

    //world boundaries
    Boundary boundaries[4] =
    {
        Boundary(1, 0, 0),
        Boundary(0, 1, 0),
        Boundary(-1, 0, -viewWidth),
        Boundary(0, -1, -viewHeight)
    };
    bool activeSpout = false;

    //Particle references for different functions
    int totalParticles = 0;
    Particle particles[maxParticles];
    Springs neighbours[maxParticles];
    Vec2 prevPos[maxParticles];
    ParticleType particleTypes[maxParticles];
    Vec4 savedParticleColors[maxParticles];


    inline int GetIndex(int x, int y)
    {
        return y*mapWidth+x;
    }



    inline int GetMapCoord(int x, int i)
    {
        if(i==0)
            return x;
        else if (i==1)
            return maxParticles+x;
        else
        {
            printf("Wrong index of map Coord: %i", x);
            return 0;
        }
    }

    Particle** map;
    const int maxParticleDouble=maxParticles*2;
    int* mapCoords;//[maxParticles][2];

    Vec2 gravity=Vec2(0.f, -9.0f);
    int delay = 0;


    void updateMap();
    void storeNeighbors();
    void applyGravity();
    void advance();
    void applyViscosity();
    void adjustSprings();
    void doubleDensityRelaxation();
    void computeNextVelocity();
    void resolveCollisions();
    void adjustColor();
    void generateParticles();
    bool addParticles(int x, int y);

    bool Update();
    void applyTemp();
};

#endif // SPH_SOLVER_HPP
