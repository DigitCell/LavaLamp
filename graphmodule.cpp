#include "graphmodule.hpp"

GraphModule::GraphModule()
{
    if(!InitSDL())
        printf("Some problems in inint SDL");

    if(!InitImGui())
        printf("Some problems in inint ImGui");

}

bool GraphModule::InitSDL()
{
    // Setup SDL
    // (Some versions of SDL before <2.0.10 appears to have performance/stalling issues on a minority of Windows systems,
    // depending on whether SDL_INIT_GAMECONTROLLER is enabled or disabled.. updating to latest version of SDL is recommended!)
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return false;
    }

    const char* glsl_version = "#version 150";
   // SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
   // SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
   // SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
   // SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

   // SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

    screen = SDL_CreateWindow("SPH", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, screenWidth, screenHeight, window_flags);
    gContext = SDL_GL_CreateContext(screen);

    SDL_GL_MakeCurrent(screen, gContext);

    SDL_GL_SetSwapInterval(1);

    //Main loop flag
    bool quit = false;
    SDL_Event e;
    SDL_StartTextInput();


    return true;
}

bool GraphModule::InitImGui()
{
    // Setup Dear ImGui context
       IMGUI_CHECKVERSION();
       ImGui::CreateContext();
       ImGui::LoadIniSettingsFromDisk("tempImgui.ini");
       ImGuiIO& io = ImGui::GetIO(); (void)io;

       io.WantCaptureMouse=true;
       //io.WantCaptureKeyboard=false;
       //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
       //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

       // Setup Dear ImGui style
       ImGui::StyleColorsDark();
       //ImGui::StyleColorsClassic();

       // Setup Platform/Renderer backends
       ImGui_ImplSDL2_InitForOpenGL(screen, gContext);
       ImGui_ImplOpenGL2_Init();

       return true;


}



//OpenGL operations. Because I'm just coloring in points, I don't need to deal with fragment shaders
void GraphModule::Render(Cuda_solver& sph_solver){
    glClearColor(0.5f, 0.5f, 0.5f, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, viewWidth, 0, viewHeight, 0, 1);

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    sph_solver.adjustColor();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glPointSize(drawRatio*particleRadius*screenWidth / viewWidth);

    glColorPointer(4, GL_FLOAT, sizeof(Vec4), sph_solver.h_savedParticleColors);
    glVertexPointer(2, GL_FLOAT, sizeof(Particle), sph_solver.h_particles);
    glDrawArrays(GL_POINTS, 0, sph_solver.h_params->totalParticles);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}


void GraphModule::GuiRender(Cuda_solver& sph_solver)
{
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    {
        ImGui::Begin("SPH particles parameters");
            ImGui::Text("number particles: %i", sph_solver.h_params->totalParticles);
            // ImGui::Checkbox("Run simulation", &show_demo_window);
            /*
            float  yield=0.08f;
            float  stiffness= 0.18f;
            float  nearStiffness= 0.01f;
            float  linearViscocity =0.5f;
            float  quadraticViscocity= 1.f;
            */

            ImGui::SliderFloat("yield", &sph_solver.h_params->yield, 0.0f, 0.3f);
            ImGui::SliderFloat("stiffness", &sph_solver.h_params->stiffness, 0.0f, 0.5f);
            ImGui::SliderFloat("nearStiffness", &sph_solver.h_params->nearStiffness, 0.0f, 0.15f);
            ImGui::SliderFloat("linearViscocity", &sph_solver.h_params->linearViscocity, 0.0f, 27.5f);
            ImGui::SliderFloat("quadraticViscocity", &sph_solver.h_params->quadraticViscocity, 0.0f, 27.5f);

            ImGui::SliderFloat("temp increase", &sph_solver.h_params->temp_increase, 0.0f, 0.15f);
            ImGui::SliderFloat("temp decrease", &sph_solver.h_params->temp_decrease, 0.0f, 0.15f);
            ImGui::SliderFloat("gravity max", &sph_solver.h_params->gravity_coeff_max, 0.7f, 111.0f);
            ImGui::SliderFloat("gravity min", &sph_solver.h_params->gravity_coeff_min, 0.0f, 10.0f);
        ImGui::End();
    }

    // Rendering
    ImGui::Render();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);

}

void GraphModule::ClearScreen()
{

}

void GraphModule::CloseRender()
{
    //close program, return true
    SDL_StopTextInput();
    SDL_DestroyWindow(screen);
    screen = NULL;
    SDL_Quit();

}

void  GraphModule::HandleEvents(SDL_Event e, Cuda_solver& sph_solver){
    if (e.type == SDL_KEYDOWN){
        if (e.key.keysym.sym == SDLK_DOWN){
            sph_solver.gravity.x = 0.f;
            sph_solver.gravity.y = -9.8f;
        }
        else if (e.key.keysym.sym == SDLK_UP){
            sph_solver.gravity.x = 0.f;
            sph_solver.gravity.y = 9.8f;
        }
        else if (e.key.keysym.sym == SDLK_LEFT){
            sph_solver.gravity.x = -9.8f;
            sph_solver.gravity.y = 0.f;
        }
        else if (e.key.keysym.sym == SDLK_RIGHT){
            sph_solver.gravity.x = 9.8f;
            sph_solver.gravity.y = 0.f;
        }
        else if (e.key.keysym.sym == SDLK_SPACE){
            sph_solver.gravity.x = 0.f;
            sph_solver.gravity.y = 0.f;
        }
        else if (e.key.keysym.sym == SDLK_LSHIFT){
            if (sph_solver.sources[2].pt == greenParticle){
                printf("1");
                sph_solver.sources[2].pt = blueParticle;
            }
            else if (sph_solver.sources[2].pt == redParticle){
                printf("2");
                sph_solver.sources[2].pt = greenParticle;
            }
        }
        else if (e.key.keysym.sym == SDLK_RSHIFT){
            if (sph_solver.sources[2].pt == greenParticle){
                printf("3");
                sph_solver.sources[2].pt = redParticle;
            }
            else if (sph_solver.sources[2].pt == blueParticle){
                printf("4");
                sph_solver.sources[2].pt = greenParticle;
            }
        }
    }
    /*
    if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT){
        activeSpout = true;
    }
    else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT){
        activeSpout = false;
    }
    */
}
