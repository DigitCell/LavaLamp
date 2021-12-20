#ifndef GRAPHMODULE_HPP
#define GRAPHMODULE_HPP

#pragma once

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl2.h"

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL/SDL_opengles2.h>
#else
#include <SDL/SDL_opengl.h>
#endif

//#include <GL/glew.h>
#include <SDL2/SDL.h>
#include "sph_solver.hpp"
#include "cuda_solver.cuh"



class GraphModule
{
public:
    GraphModule();

    bool InitSDL();
    bool InitImGui();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    float f = 0.0f;
    int counter = 0;



    SDL_Window* screen = NULL;
    SDL_GLContext gContext;

    SDL_Event event;


    void HandleEvents(SDL_Event e, Cuda_solver& sph_solver);
    void Render(Cuda_solver &sph_solver);
    void GuiRender(Cuda_solver &sph_solver);

    void ClearScreen();

    void CloseRender();


};

#endif // GRAPHMODULE_HPP
