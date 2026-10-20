#pragma once

/*This file defines the numerical types, geometry
choices, simulation paramters. and bead state 
shared by the rest of the program */

/*tells compiler to process this header only once per
compilatiomn unit, even if severl incided files refer to it*/

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
enum class MobilityType { OSEEN, ROTNE_PRAGER, BLAKE };

// Independent drives add forces along each target bead's local stroke axis.
// The clock is absolute simulation time; start/stop only gate the waveform.
struct Drive {
    double amplitude, period, phase, start, stop;
    int waveform; // 0 square, 1 sine, 2 constant
    std::vector<int> beads;
};

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
    // Fixed local oscillation directions, in radians; legacy inputs use zero.
    std::vector<double> angles;
    std::vector<int> signal_beads; /*allows for signals on each bead*/

    std::vector<Drive> drives;
    bool explicit_drives = false; // Empty explicit list means no external forcing.
    bool direct_traps = false; // H=1; avoids bead-index-dependent strengths in circuits.
    long long seed = -1; // -1 retains the original random initialisation.
    GeometryType geometry; /*remember arrangement choice we make*/
    MobilityType mobility = MobilityType::ROTNE_PRAGER;
    // Fixed bead-centre height above z=0. Used only by Blake mode.
    double wall_height = 6.0;
    // Wall-compatible imposed shear: u(z)=z*(shear_x,shear_y,0).
    double shear_x = 0.0, shear_y = 0.0;
    // Scalars above are defaults; these resolved arrays govern each trap.
    std::vector<double> alpha_per_bead, beta_per_bead;
    std::vector<double> amplitude_per_bead, epsilon_per_bead;
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
