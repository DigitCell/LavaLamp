#ifndef MAINLOOP_HPP
#define MAINLOOP_HPP


#pragma once
#include "graphmodule.hpp"
#include "Constants.hpp"
#include "cuda_solver.cuh"

class MainLoop
{
public:
    MainLoop();

    GraphModule graphModule;
    Cuda_solver cuda_solver;

    bool RunLoop();

};

#endif // MAINLOOP_HPP
