#pragma once

/*making the necessary definitions avaliable*/
#include "Types.h"
#include "PhysicsEngine.h" /*will use this to calculate forces, mobility, and thermal-noise factors*/
#include <fstream> /* provides file strems for reading inputs and writing results*/
#include <memory> /*provides std::unique_ptr, which allows simulation to own its physics engine 
                    and manage its lifetime*/

namespace modern {

/*'class' is 'struct' nut we can demand what is private and public*/
class Simulation {
/*public: declaration below this lavel can be accessed from outside this class. They define how 
main.cpp interacts with the simulation.

A constructor prepares a newly created object. I has the same name as its class and no return type.
It accepts three arguments: 'input_file': contains simulation parameters, 'geometry': chain, ring or 
arbitrary, 'use_rotne_prager': whether to use rotne prager rather than Oseen mobility

Basically the public face says: 'provide an input and model choices, then run the experiment*/
public:
    explicit Simulation(const std::string& input_file, GeometryType geometry, bool use_rotne_prager);
    
    void run();

/*priavte face does the following: 'parseInput': translate file contents into the config, 
'initializeState': constructs starting coords and trap states. 'step': advance positions by one 
numerical time step, 'activeTrapping: apply the position-dependent switching rule*/
private:
    void parseInput(const std::string& input_file);
    void initializeState();
    void step(double time);
    void activeTrapping();
    /*'saveData': saves bead trajectory, 'saveTraps': saves trap centres and switching history, 
    'saveForces': store force values, 'generateTitle': output naming*/
    void saveData(std::ostream& out, int frame);
    void saveTraps(std::ostream& out, int frame);
    void saveForces(std::ostream& out, int frame);
    
    std::string generateTitle() const;

    /*Keeping everything needed for the run together*/
    std::string input_filename_;
    SimulationConfig config_;
    State state_;
    std::unique_ptr<PhysicsEngine> engine_;
    
    /*keeping these arrays as members lets the program allocate their main storage during 
    setup and reuse it, rather than recreating every buffer from scratch at each step*/
    MatrixXd tensor_; // holds: mobility matrix, needed: converts bead forces into velocities
    MatrixXd diffusion_; // holds: cholesky factor L*L^T = K_b*T*H, needed: transform independent gaussian samples into correlated displacements
    VectorXd forces_; // holds: applied force vector, needed: for H*F
    VectorXd noise_; // holds: thermal displacement vector, needed: add stochastic contribution to position update
    
    GeometryType geometry_arg_;
};

} // namespace modern
