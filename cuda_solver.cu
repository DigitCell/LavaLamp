#include "cuda_solver.cuh"

__constant__  gpuParams params;

__device__ int GetIndex(int x, int y)
{
    return y*params.mapWidth+x;
}
__device__ int GetIndexMap(int i,int x, int y)
{
    return  i*params.maxMapCoeff1+y*params.mapWidth+x;
}

__device__ int GetIndexNeightb(int i, int j)
{
    return  i*maxSprings+j;
}

Cuda_solver::Cuda_solver()
{

         CudaInit();
        // params init

        h_params=new gpuParams;

        h_params->yield=0.08f;
        h_params->stiffness= 0.18f;
        h_params->nearStiffness= 0.01f;
        h_params->linearViscocity =0.5f;
        h_params->quadraticViscocity= 1.f;

        h_params->temp_increase=0.05f;
        h_params->temp_decrease=0.005f;
        h_params->gravity_coeff_max=11.0f;
        h_params->gravity_coeff_min=3.0f;

        h_params->totalParticles=0;

        gravity.x=0.0f;
        gravity.y=-9.8f;
        h_params->gravity.x=gravity.x;
        h_params->gravity.y=gravity.y;

        h_params->maxMapCoeff1=mapWidth*mapHeight;
        h_params->maxMapCoeff2=mapWidth*mapHeight;

        h_params->mapWidth=mapWidth;
        h_params->mapHeight=mapHeight;


        h_particles=(Particle*)malloc(sizeof(Particle)*maxParticles);
        //h_neighbours=(Springs*)malloc(sizeof(Springs)*maxParticles);
        h_prevPos=(float2*)malloc(sizeof(float2)*maxParticles);

        h_particleTypes=(ParticleType*)malloc(sizeof(ParticleType)*maxParticles);
        h_savedParticleColors=(Vec4*)malloc(sizeof(Vec4)*maxParticles);



        h_map=(int*)malloc(sizeof(int)*mapWidth*mapHeight*maxParticles);
        h_map_size=(int*)malloc(sizeof(int)*mapWidth*mapHeight);

        h_mapCoords=(int2*)malloc(sizeof(int2)*maxParticles);
        h_boundaries=(float3*)malloc(sizeof(float3)*4);

        h_neightb_size=(int*)malloc(sizeof(int)*maxParticles);


        checkCudaCall(cudaMalloc(&d_particles,sizeof(Particle)*maxParticles));

        checkCudaCall(cudaMalloc(&d_prevPos,sizeof(float2)*maxParticles));
        checkCudaCall(cudaMalloc(&d_particleTypes,sizeof(ParticleType)*maxParticles));
        checkCudaCall(cudaMalloc(&d_savedParticleColors,sizeof(Vec4)*maxParticles));


        checkCudaCall(cudaMalloc(&d_map,sizeof(int)*mapWidth*mapHeight*maxParticles));
        checkCudaCall(cudaMalloc(&d_map_size,sizeof(int)*mapWidth*mapHeight));

        checkCudaCall(cudaMalloc(&d_mapCoords,sizeof(int2)*maxParticles));
        checkCudaCall(cudaMalloc(&d_boundaries,sizeof(float3)*4));

        gravity=make_float2(0.f, -9.0f);

        for(int i=0;i<4;i++)
        {
            h_boundaries[i].x=boundaries[i].x;
            h_boundaries[i].y=boundaries[i].y;
            h_boundaries[i].z=boundaries[i].c;

        }

        checkCudaCall(cudaMalloc(&d_neightb_index,sizeof(int)*maxSprings*maxParticles));
        checkCudaCall(cudaMalloc(&d_neightb_size,sizeof(int)*maxParticles));
        checkCudaCall(cudaMalloc(&d_neightb_r,sizeof(float)*maxSprings*maxParticles));
        checkCudaCall(cudaMalloc(&d_neightb_Lij,sizeof(float)*maxSprings*maxParticles));



        memset(h_map, 0, mapWidth*mapHeight*sizeof(int));

        memset(h_neightb_size, 0, maxParticles* sizeof(int));

        checkCudaCall(cudaMemcpyToSymbol(params,h_params,sizeof(gpuParams)));
        UpdateGPUBuffers();
        ClearMap();
        updateMap();

}


void Cuda_solver::CudaInit()
{
    int devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           devID, _ConvertSMVer2ArchName(major, minor), major, minor);
}


void Cuda_solver::generateParticles(){
    if (h_params->totalParticles == maxParticles)
        return;

    if (delay++ < 2)
        return;

    for (int turn = 0; turn<2; turn++){
        Source& source = sources[turn];
        if (source.count >= maxParticles / 2) continue;

        for (int i = 0; i <= 2 && h_params->totalParticles<maxParticles; i++){
            Particle& p = h_particles[h_params->totalParticles];
            ParticleType& pt = h_particleTypes[h_params->totalParticles];
            h_params->totalParticles++;

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

    UpdateGPUBuffers();
}

void Cuda_solver::UpdateGPUBuffers()
{
    //checkCudaCall(cudaMemcpyToSymbol(params,h_params,sizeof(gpuParams)));

    checkCudaCall(cudaMemcpy(d_boundaries,h_boundaries,sizeof(float3)*4,cudaMemcpyHostToDevice));

    checkCudaCall(cudaMemcpy(d_particles,h_particles,sizeof(Particle)*maxParticles,cudaMemcpyHostToDevice));

    checkCudaCall(cudaMemcpy(d_neightb_size,h_neightb_size,sizeof(int)*maxParticles,cudaMemcpyHostToDevice));
    //checkCudaCall(cudaMemcpy(d_neighbours,h_neighbours,sizeof(Springs)*maxParticles,cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_prevPos,h_prevPos,sizeof(float2)*maxParticles,cudaMemcpyHostToDevice));

    checkCudaCall(cudaMemcpy(d_particleTypes,h_particleTypes,sizeof(ParticleType)*maxParticles,cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_savedParticleColors,h_savedParticleColors,sizeof(int4)*maxParticles,cudaMemcpyHostToDevice));

    //checkCudaCall(cudaMemcpy(d_map,h_map,sizeof(int)*mapWidth*mapHeight,cudaMemcpyHostToDevice));
    //checkCudaCall(cudaMemcpy(d_mapCoords,h_mapCoords,sizeof(int2)*mapWidth*mapHeight,cudaMemcpyHostToDevice));

}

void Cuda_solver::UpdateHostBuffers()
{

    checkCudaCall(cudaMemcpy(h_particles,d_particles,sizeof(Particle)*maxParticles,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_neighbours,d_neighbours,sizeof(Springs)*maxParticles,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_prevPos,d_prevPos,sizeof(float2)*maxParticles,cudaMemcpyDeviceToHost));

  //  checkCudaCall(cudaMemcpy(h_particleTypes,d_particleTypes,sizeof(ParticleType)*maxParticles,cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(h_savedParticleColors,d_savedParticleColors,sizeof(int4)*maxParticles,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_map_size,d_map_size,sizeof(int)*mapWidth*mapHeight,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_map,d_map,sizeof(int)*mapWidth*mapHeight*maxParticles,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_mapCoords,d_mapCoords,sizeof(int*)*mapWidth*mapHeight,cudaMemcpyDeviceToHost));

}



//2-Dimensional gravity for player input
void Cuda_solver::applyTemp(){

    if(h_params->totalParticles>1)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaApplyTemp<<<blocksPerGrid, threadsPerBlock>>>(d_particles);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaApplyTemp(Particle* particles)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];
        if(p.posY<1.95 and p.posY>0.25)
            p.temp+=params.temp_increase*(1.0-p.posY/2.0);
        else
            p.temp-=params.temp_decrease;
        if(p.temp>params.gravity_coeff_max)
            p.temp=params.gravity_coeff_max;
        if(p.temp<params.gravity_coeff_min)
            p.temp=params.gravity_coeff_min;

    }
}

//2-Dimensional gravity for player input
void Cuda_solver::applyGravity()
{
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaApplyGravity<<<blocksPerGrid, threadsPerBlock>>>(d_particles);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaApplyGravity(Particle* particles)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

     //   if(i==0)
     //        printf("coord %f %f",p.posX, p.posY);

        p.velY += params.gravity.y*deltaTime+p.temp*deltaTime;
        p.velX += params.gravity.x*deltaTime;

    }
}

//applies viscosity impulses to particles
void Cuda_solver::applyViscosity()
{
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaApplyViscosity<<<blocksPerGrid, threadsPerBlock>>>(d_particles,
                                                               d_neightb_index, d_neightb_size, d_neightb_r, d_neightb_Lij);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaApplyViscosity(Particle* particles,
                                   int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        for (int j = 0; j < neightb_size[i]; j++){
            const Particle& pNear = particles[neightb_index[GetIndexNeightb(i,j)]];

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

                float I = 0.5f * deltaTime * a * (params.linearViscocity*u + params.quadraticViscocity*u*u);

                particles[i].velX -= I * diffX;
                particles[i].velY -= I * diffY;
            }
        }

    }
}

//Advances particle along its velocity
void Cuda_solver::advance()
{
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaAdvance<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_prevPos);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaAdvance(Particle* particles, float2* prevPos)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        prevPos[i].x = p.posX;
        prevPos[i].y = p.posY;

        p.posX += deltaTime * p.velX;
        p.posY += deltaTime * p.velY;

    }
}


void Cuda_solver::adjustSprings()
{
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaAdjustSprings<<<blocksPerGrid, threadsPerBlock>>>(d_particles,
                                                              d_neightb_index, d_neightb_size, d_neightb_r, d_neightb_Lij);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaAdjustSprings(Particle* particles,
                                  int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        const Particle& p = particles[i];
        //iterate through that particles neighbors
        for (int j = 0; j < neightb_size[i]; j++){
            const Particle& pNear =  particles[neightb_index[GetIndexNeightb(i,j)]];

            float r = neightb_r[GetIndexNeightb(i,j)];
            float q = r / particleHeight;

            if (q < 1 && q > 0.0000000000001f){
                float d = params.yield*neightb_Lij[GetIndexNeightb(i,j)];

                //calculate spring values
                if (r>particleHeight + d){
                    neightb_Lij[GetIndexNeightb(i,j)]+= deltaTime*alphaSpring*(r - particleHeight - d);
                }
                else if (r<particleHeight - d){
                    neightb_Lij[GetIndexNeightb(i,j)] -= deltaTime*alphaSpring*(particleHeight - d - r);
                }

                //apply those changes to the particle
                float Lij = neightb_Lij[GetIndexNeightb(i,j)];
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



//Resets the map of the scene, re-adding every particle based on where it is at this moment
void Cuda_solver::updateMap()
{
    ClearMap();

    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;


        CudaUpdateMap<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_map, d_map_size, d_mapCoords);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaUpdateMap(Particle* particles, int* map, int *map_size, int2* mapCoords)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];
        int x = p.posX / particleHeight;
        int y = p.posY / particleHeight;

        if (x < 1) x = 1;
        else if (x > params.mapWidth - 2) x = params.mapWidth - 2;

        if (y < 1)
            y = 1;
        else if (y > params.mapHeight - 2)
            y = params.mapHeight - 2;

        //this handles the linked list between particles on the same square

        int& indexAdd=map_size[GetIndex(x,y)];
/*
        if(indexAdd>0)
        {
           printf(" PrePart %i, mapsize %i , x=%i, y=%i" ,i, indexAdd,x,y );
        }
*/
        map[GetIndexMap(atomicAdd(&indexAdd,1),x,y)] = i;
       // int oldIndex =atomicAdd(&indexAdd,1);

        mapCoords[i].x = x;
        mapCoords[i].y = y;

    }
}


//Resets the map of the scene, re-adding every particle based on where it is at this moment
void Cuda_solver::ClearMap()
{

    int threadsPerBlock = 256;
    int blocksPerGrid =
            (h_params->maxMapCoeff1 + threadsPerBlock - 1) / threadsPerBlock;

    CudaClearMap<<<blocksPerGrid, threadsPerBlock>>>(d_map, d_map_size);
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");
}

__global__ void CudaClearMap(int* map, int *map_size)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.maxMapCoeff1)
    {
        map_size[i]=0;
        if(map_size[i]>0)
           printf("not Clear mapsize %i" , map_size[i] );
    }
}


//saves neighbors for lookup in other functions
void Cuda_solver::storeNeighbors(){
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaStoreNeighbors<<<blocksPerGrid, threadsPerBlock>>>(d_particles,  d_map, d_map_size, d_mapCoords,
                                                               d_neightb_index, d_neightb_size, d_neightb_r, d_neightb_Lij);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaStoreNeighbors(Particle* particles,  int* map, int *map_size, int2* mapCoords,
                                   int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];
        int pX = mapCoords[i].x;
        int pY = mapCoords[i].y;

        neightb_size[i]=0;

        //iterate over the nine squares on the grid around p
        for (int mapX = pX - 1; mapX <= pX + 1; mapX++){
            for (int mapY = pY - 1; mapY <= pY + 1; mapY++){
                //go through the current square's linked list of overlapping values, if there is one
               // for (Particle* nextDoor = &particles[map[GetIndex(mapX,mapY)]]; nextDoor != NULL; nextDoor = nextDoor->next){

                if(mapX<0 or mapY<0 or mapX>params.mapWidth-1 or mapY>params.mapHeight-1)
                     continue;
                if(map_size[GetIndex(mapX,mapY)]==0)
                    continue;
                for(int ip=0; ip<map_size[GetIndex(mapX,mapY)];ip++)
                {
                    const Particle& pj =  particles[map[GetIndexMap(ip,mapX,mapY)]];

                    float diffX = pj.posX - p.posX;
                    float diffY = pj.posY - p.posY;
                    float r2 = diffX*diffX + diffY*diffY;
                    float r = sqrt(r2);
                    float q = r / particleHeight;

                    //save this neighbor
                    if (q < 1 && q > 0.0000000000001f){

                        const int j=neightb_size[i];// (neightb_size[i]==0)?0:neightb_size[i]-1;
                        if (neightb_size[i] < maxSprings){
                            neightb_index[GetIndexNeightb(i, j)]=map[GetIndexMap(ip,mapX,mapY)];
                            neightb_r[GetIndexNeightb(i,j)]=r;
                            neightb_Lij[GetIndexNeightb(i,j)]=particleHeight;
                            neightb_size[i]++;
                        }

                    }
                }
            }
        }

    }
}


//This maps pretty closely to the outline in the paper. Find density and pressure for all particles,
//then apply a displacement based on that. There is an added if statement to handle surface tension for multiple weights of particles
void Cuda_solver::doubleDensityRelaxation(){
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaDoubleDensityRelaxation<<<blocksPerGrid, threadsPerBlock>>>(d_particles,
                                                                        d_neightb_index, d_neightb_size, d_neightb_r, d_neightb_Lij);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaDoubleDensityRelaxation(Particle* particles,
                                            int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        float density = 0;
        float nearDensity = 0;

        for (int j = 0; j < neightb_size[i]; j++){
            const Particle& pNear = particles[neightb_index[GetIndexNeightb(i,j)]];// *neighbours[i].particles[j];

            float r = neightb_r[GetIndexNeightb(i,j)];
            float q = r / particleHeight;
            if (q>1) continue;
            float a = 1 - q;

            density += a*a * pNear.m * 20;
            nearDensity += a*a*a * pNear.m * 30;
        }
        p.pressure = params.stiffness * (density - p.m*restDensity);
        p.nearPressure = params.nearStiffness * nearDensity;
        float dx = 0, dy = 0;

        for (int j = 0; j < neightb_size[i]; j++){
            const Particle& pNear = particles[neightb_index[GetIndexNeightb(i,j)]];

            float diffX = pNear.posX - p.posX;
            float diffY = pNear.posY - p.posY;

            float r = neightb_r[GetIndexNeightb(i,j)];
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

void Cuda_solver::computeNextVelocity(){
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaComputeNextVelocity<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_prevPos);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaComputeNextVelocity(Particle* particles, float2* prevPos)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];
        p.velX = (p.posX - prevPos[i].x) / deltaTime;
        p.velY = (p.posY - prevPos[i].y) / deltaTime;

    }
}


//Only checks if particles have left window, and push them back if so
void Cuda_solver::resolveCollisions(){
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaResolveCollisions<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_boundaries);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaResolveCollisions(Particle* particles, float3* boundaries)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        for (int j = 0; j<4; j++){
            const float3& boundary = boundaries[j];
            float distance = boundary.x*p.posX + boundary.y*p.posY - boundary.z;

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
void Cuda_solver::adjustColor()
{
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaAdjustColor<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_savedParticleColors, d_particleTypes);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaAdjustColor(Particle* particles, Vec4* savedParticleColors, ParticleType* particleTypes)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
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
//Runs through all of the logic 7 times a frame
bool Cuda_solver::Update()
{

    checkCudaCall(cudaMemcpyToSymbol(params,h_params,sizeof(gpuParams)));
    //for (int step = 0; step<timeStep; step++)
    for (int step = 0; step<2; step++)
    {

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

    UpdateHostBuffers();
    //checkCudaCall(cudaMemcpyFromSymbol(h_params,params,sizeof(gpuParams)));

   // UpdateHostBuffers();
    /*
    ClearMap();

    for(int i=0; i<h_params->maxMapCoeff1;i++)
    {
        if(h_map_size[i]>0)
            printf(" %i", h_map_size[i]);
    }
*/
/*
    printf(" Start print \n");
    for(int ix=0; ix<h_params->mapWidth;ix++)
    {
        for(int iy=0; iy<h_params->mapHeight;iy++)
        {
            if(h_map_size[iy*mapWidth+ix]>0)
            {
                printf(" %i \n", h_map_size[iy*mapWidth+ix]);

                for(int ii=0; ii<h_map_size[iy*mapWidth+ix]; ii++)
                {
                     printf("ii %i ", h_map[ ii*h_params->maxMapCoeff1+iy*h_params->mapWidth+ix]);
                }
                printf("\n");
            }
        }
    }
    */
    return true;
}
