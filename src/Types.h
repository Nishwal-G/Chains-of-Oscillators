/*This file defines the numerical types, geometry
choices, simulation paramters. and bead state 
shared by the rest of the program */

/*tells compiler to process this header only once per
compilatiomn unit, even if severl incided files refer to it*/
#pragma once

/*we are importing all necessary definitions*/
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace modern {

/*introduce aliases for all eigen types
to make life easier in the future. Xd 
means the size is chosen at teh runtime,
so input array length doesn't require change 
in source code.*/
using Vector2d = Eigen::Vector2d;
using Matrix2d = Eigen::Matrix2d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;


/*Here we define the allowed geometry labels.
RING places reference beads in a circle, CHAIN
along a line and ARBITRARY according to supplied
coords.*/
enum class GeometryType {
    RING,
    CHAIN,
    ARBITRARY
};

/*Here we are effecitvley building a 'class'
or 'structure' where we compile many arguments
into a single place to make it easier to pass in
the future.*/
struct SimulationConfig {
    /*scalar members*/
    int num_beads; /*Number of beads in array*/
    double amplitude; /*switching excursion paramter*/
    double base_distance; /*reference spacing scale to normalise trap strength*/
    double bead_radius; /*Bead radius for Stokes Drag and Mobility calcs*/
    double dt; /*timestep*/
    double total_time; /*simulation duration*/
    double viscosity;
    double alpha; /*active x-potential exponent*/
    double kx; /*default x-potential coefficient*/
    double beta; /*transverse y-potential exponent*/
    double ky; /*default y-potential coefficient*/
    double temperature; /*thermal noise*/
    int sampling_feedback; /*steps between switching checks*/
    int sampling_write; /*steps between saved records*/
    double epsilon;/* used to trap centre beyond threhold*/
    double velox; /*background flow component x*/
    double ks;  /*not used*/
    double fl0; /*not used*/
    double veloy; /*background flow component x*/
    double f0_signal; /*signal force amplitude*/
    double t_signal; /*signal period*/
    double default_lambda; /*fill missing per-bead lambda values*/
    
    /*vector members: allows us to be nonuniform in our paramters*/
    std::vector<double> inter_bead_distances; /*allow nonuniform chain spacing*/
    std::vector<double> lambdas; /*allow each to bead to have its own trap-strength profile*/
    std::vector<Vector2d> initial_positions; /*allow for ARBITRARY 'reference' positions*/
    std::vector<double> kx_per_bead; /*allow for nonuniform kx*/
    std::vector<double> ky_per_bead; /*allow for nonuniform ky*/
    std::vector<int> signal_beads; /*allows for signals on each bead*/

    GeometryType geometry; /*remember arrangement choice we make*/
    bool use_rotne_prager = true;/*rotne-prager mobility calc as default, can overwrite 
                                 to oseen in command line*/
};

/*this structure stores the current state of the system, 
while the config above gave it the intial conditions*/
struct State {
    VectorXd positions; /*where the beads currently are*/
    VectorXd trap_offsets; /*where their oscillation centres are*/
    VectorXd reference_pos; /*where the active traps lie relative to those references*/

    /*recall we need r-r_ref for switching and r-r_trap for force*/

    /* create three arrays large enough to hold bead and trap coords, 
    then fill them with zero*/
    explicit State(int n)
        : positions(2 * n), trap_offsets(2 * n), reference_pos(2 * n) {
        positions.setZero();
        trap_offsets.setZero();
        reference_pos.setZero();
    }
};

} // namespace modern
