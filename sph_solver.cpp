#include "sph_solver.hpp"

SPH_solver::SPH_solver()
{

    map=(Particle**)malloc(sizeof (Particle*)*mapWidth*mapHeight);
    //mapCoords=new int(maxParticles*2);
    mapCoords=(int*)malloc(sizeof(int)*maxParticles*2);

    memset(particles, 0, maxParticles*sizeof(Particle));
    updateMap();

}



//Resets the map of the scene, re-adding every particle based on where it is at this moment
void SPH_solver::updateMap()
{
    memset(map, 0, mapWidth*mapHeight*sizeof(Particle*));

    for (int i = 0; i<totalParticles; i++)
    {
        Particle& p = particles[i];
        int x = p.posX / particleHeight;
        int y = p.posY / particleHeight;

        if (x < 1) x = 1;
        else if (x > mapWidth - 2) x = mapWidth - 2;

        if (y < 1)
            y = 1;
        else if (y > mapHeight - 2)
            y = mapHeight - 2;

        //this handles the linked list between particles on the same square
        p.next = map[GetIndex(x,y)];
        map[GetIndex(x,y)] = &p;

        mapCoords[GetMapCoord(i,0)] = x;
        mapCoords[GetMapCoord(i,1)] = y;
    }
}

//saves neighbors for lookup in other functions
void SPH_solver::storeNeighbors(){
    for (int i = 0; i<totalParticles; i++){
        Particle& p = particles[i];
        int pX = mapCoords[GetMapCoord(i,0)];
        int pY = mapCoords[GetMapCoord(i,1)];

        neighbours[i].count = 0;

        //iterate over the nine squares on the grid around p
        for (int mapX = pX - 1; mapX <= pX + 1; mapX++){
            for (int mapY = pY - 1; mapY <= pY + 1; mapY++){
                //go through the current square's linked list of overlapping values, if there is one
                for (Particle* nextDoor = map[GetIndex(mapX,mapY)]; nextDoor != NULL; nextDoor = nextDoor->next){
                    const Particle& pj = *nextDoor;

                    float diffX = pj.posX - p.posX;
                    float diffY = pj.posY - p.posY;
                    float r2 = diffX*diffX + diffY*diffY;
                    float r = sqrt(r2);
                    float q = r / particleHeight;

                    //save this neighbor
                    if (q < 1 && q > 0.0000000000001f){
                        if (neighbours[i].count < maxSprings){
                            neighbours[i].particles[neighbours[i].count] = &pj;
                            neighbours[i].r[neighbours[i].count] = r;
                            neighbours[i].Lij[neighbours[i].count] = particleHeight;
                            neighbours[i].count++;
                        }
                    }
                }
            }
        }
    }
}

//2-Dimensional gravity for player input
void SPH_solver::applyGravity(){
    for (int i = 0; i<totalParticles; i++){
        Particle& p = particles[i];
        p.velY += gravity.y*deltaTime+p.temp*deltaTime;
        p.velX += gravity.x*deltaTime;
    }
}

//2-Dimensional gravity for player input
void SPH_solver::applyTemp(){
    for (int i = 0; i<totalParticles; i++){
        Particle& p = particles[i];
        if(p.posY<0.95 and p.posY>0.25)
            p.temp+=temp_increase*(1.0-p.posY);
        else
            p.temp-=temp_decrease;
        if(p.temp>gravity_coeff_max)
            p.temp=gravity_coeff_max;
        if(p.temp<gravity_coeff_min)
            p.temp=gravity_coeff_min;
    }
}

//Advances particle along its velocity
void SPH_solver::advance(){
    for (int i = 0; i<totalParticles; i++){
        Particle& p = particles[i];

        prevPos[i].x = p.posX;
        prevPos[i].y = p.posY;

        p.posX += deltaTime * p.velX;
        p.posY += deltaTime * p.velY;
    }
}

//applies viscosity impulses to particles
void SPH_solver::applyViscosity(){
    //cycle through all particles
    for (int i = 0; i<totalParticles; i++){
        Particle& p = particles[i];

        for (int j = 0; j < neighbours[i].count; j++){
            const Particle& pNear = *neighbours[i].particles[j];

            float diffX = pNear.posX - p.posX;
            float diffY = pNear.posY - p.posY;

            float r2 = diffX*diffX + diffY*diffY;
            float r = sqrt(r2);

            float q = r / particleHeight;

            if (q>1) continue;

            float diffVelX = p.velX - pNear.velX;
            float diffVelY = p.velY - pNear.velY;
            float u = diffVelX*diffX + diffVelY*diffY;

            if (u > 0){
                float a = 1 - q;
                diffX /= r;
                diffY /= r;
                u /= r;

                float I = 0.5f * deltaTime * a * (linearViscocity*u + quadraticViscocity*u*u);

                particles[i].velX -= I * diffX;
                particles[i].velY -= I * diffY;
            }
        }
    }
}

//This combines both algorithms for spring adjustment and displacement.
void SPH_solver::adjustSprings(){
    //iterate through all particles
    for (int i = 0; i < totalParticles; i++)	{
        const Particle& p = particles[i];
        //iterate through that particles neighbors
        for (int j = 0; j < neighbours[i].count; j++){
            const Particle& pNear = *neighbours[i].particles[j];

            float r = neighbours[i].r[j];
            float q = r / particleHeight;

            if (q < 1 && q > 0.0000000000001f){
                float d = yield*neighbours[i].Lij[j];

                //calculate spring values
                if (r>particleHeight + d){
                    neighbours[i].Lij[j] += deltaTime*alphaSpring*(r - particleHeight - d);
                }
                else if (r<particleHeight - d){
                    neighbours[i].Lij[j] -= deltaTime*alphaSpring*(particleHeight - d - r);
                }

                //apply those changes to the particle
                float Lij = neighbours[i].Lij[j];
                float diffX = pNear.posX - p.posX;
                float diffY = pNear.posY - p.posY;
                float displaceX = deltaTime*deltaTime*kSpring*(1 - Lij / particleHeight)*(Lij - r)*diffX;
                float displaceY = deltaTime*deltaTime*kSpring*(1 - Lij / particleHeight)*(Lij - r)*diffY;
                particles[i].posX -= 0.5f*displaceX;
                particles[i].posY -= 0.5f*displaceY;
            }
        }
    }
}

//This maps pretty closely to the outline in the paper. Find density and pressure for all particles,
//then apply a displacement based on that. There is an added if statement to handle surface tension for multiple weights of particles
void SPH_solver::doubleDensityRelaxation(){
    for (int i = 0; i < totalParticles; i++)	{
        Particle& p = particles[i];

        float density = 0;
        float nearDensity = 0;

        for (int j = 0; j < neighbours[i].count; j++){
            const Particle& pNear = *neighbours[i].particles[j];

            float r = neighbours[i].r[j];
            float q = r / particleHeight;
            if (q>1) continue;
            float a = 1 - q;

            density += a*a * pNear.m * 20;
            nearDensity += a*a*a * pNear.m * 30;
        }
        p.pressure = stiffness * (density - p.m*restDensity);
        p.nearPressure = nearStiffness * nearDensity;
        float dx = 0, dy = 0;

        for (int j = 0; j < neighbours[i].count; j++){
            const Particle& pNear = *neighbours[i].particles[j];

            float diffX = pNear.posX - p.posX;
            float diffY = pNear.posY - p.posY;

            float r = neighbours[i].r[j];
            float q = r / particleHeight;
            if (q>1) continue;
            float a = 1 - q;
            float d = (deltaTime*deltaTime) * ((p.nearPressure + pNear.nearPressure)*a*a*a*53 + (p.pressure + pNear.pressure)*a*a*35) / 2;

            // weight is added to the denominator to reduce the change in dx based on its weight
            dx -= d * diffX / (r*p.m);
            dy -= d * diffY / (r*p.m);

            //surface tension is mapped with one type of particle,
            //this allows multiple weights of particles to behave appropriately
            if (p.m == pNear.m && multipleParticleTypes == true){
                dx += surfaceTension * diffX;
                dy += surfaceTension * diffY;
            }
        }

        p.posX += dx;
        p.posY += dy;
    }
}

void SPH_solver::computeNextVelocity(){
    for (int i = 0; i<totalParticles; ++i){
        Particle& p = particles[i];
        p.velX = (p.posX - prevPos[i].x) / deltaTime;
        p.velY = (p.posY - prevPos[i].y) / deltaTime;
    }
}

//Only checks if particles have left window, and push them back if so
void SPH_solver::resolveCollisions(){
    for (int i = 0; i<totalParticles; i++){
        Particle& p = particles[i];

        for (int j = 0; j<4; j++){
            const Boundary& boundary = boundaries[j];
            float distance = boundary.x*p.posX + boundary.y*p.posY - boundary.c;

            if (distance < particleRadius){
                if (distance < 0)
                    distance = 0;
                p.velX += 0.1f*(particleRadius - distance) * boundary.x / deltaTime;
                p.velY += (particleRadius - distance) * boundary.y / deltaTime;
            }

            //The resolve collisions tends to overestimate the needed counter velocity, this limits that
            if (p.velX > 0.5) p.velX = 0.5;
            if (p.velY > 2) p.velY = 2;
            if (p.velX < -0.5) p.velX = -0.5;
            if (p.velY < -2) p.velY = -2;
        }
    }
}

//Iterates through every particle and multiplies its RGB values based on speed.
//speed^2 is just used to make the difference in speeds more noticeable.
void SPH_solver::adjustColor(){
    for (int i = 0; i<totalParticles; i++){
        const Particle& p = particles[i];
        const ParticleType& pt = particleTypes[i];

        float speed2 = p.temp/10.f;//.velX*p.velX + p.velY*p.velY;

        Vec4& color = savedParticleColors[i];
        color = pt.color;
        color.r *= 0.5f + velocityFactor*speed2;
        color.g *= 0.5f + velocityFactor*speed2;
        color.b *= 0.5f + velocityFactor*speed2;
    }
}

void SPH_solver::generateParticles(){
    if (totalParticles == maxParticles)
        return;

    if (delay++ < 2)
        return;

    for (int turn = 0; turn<2; turn++){
        Source& source = sources[turn];
        if (source.count >= maxParticles / 2) continue;

        for (int i = 0; i <= 2 && totalParticles<maxParticles; i++){
            Particle& p = particles[totalParticles];
            ParticleType& pt = particleTypes[totalParticles];
            totalParticles++;

            source.count++;

            //for an even distribution of particles
            float offset = float(i) / 1.5f;
            offset *= 0.2f;
            p.posX = source.position.x - offset*source.direction.y;
            p.posY = source.position.y + offset*source.direction.x;
            p.velX = source.speed *source.direction.x;
            p.velY = source.speed *source.direction.y;
            p.m = source.pt.mass;
            p.temp=0;

            pt = source.pt;
        }
    }
    delay = 0;
}

//Mouse particle generator. generates less particles, but does it more consistently.
//It can also only be run every frame, while generate can run 7 times a frame
bool SPH_solver::addParticles(int x, int y){
    if (totalParticles == maxParticles)
        return false;

    float localX = (viewWidth * (float(x) / screenWidth)),
        localY = (viewHeight * (float(screenHeight - y) / screenHeight));
    sources[2].position = Vec2(localX, localY);

    for (int i = 0; i <= 2 && totalParticles<maxParticles; i++){
        Particle& p = particles[totalParticles];
        ParticleType& pt = particleTypes[totalParticles];
        totalParticles++;

        sources[2].count++;

        p.posX = sources[2].position.x + (particleHeight / 2)*i;
        p.posY = sources[2].position.y;
        p.velX = sources[2].speed * sources[2].direction.x;
        p.velY = sources[2].speed * sources[2].direction.y;
        p.m = sources[2].pt.mass;

        p.temp=0;

        pt = sources[2].pt;
    }

    return true;
}

//Runs through all of the logic 7 times a frame
bool SPH_solver::Update()
{
    for (int step = 0; step<timeStep; step++){
        generateParticles();

        applyTemp();

        applyGravity();
        applyViscosity();
        advance();
        adjustSprings();
        updateMap();
        storeNeighbors();
        doubleDensityRelaxation();
        computeNextVelocity();
        resolveCollisions();
    }
    return true;
}
