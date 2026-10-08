#pragma once

/*
PhysicsEngine performs the physical calculations.
Simulation decides when to call these calculations and uses
their results to advance the bead positions.
*/

// Provides SimulationConfig, State, and the Eigen vector/matrix aliases.
#include "Types.h"

// Provides random-number generators and probability distributions.
#include <random>

// Groups these definitions under the name modern.
// The full class name is modern::PhysicsEngine.
namespace modern {

class PhysicsEngine {
public:
    /*
    Constructor: prepares an engine using the simulation parameters.

    const SimulationConfig&:
      &     - receives the configuration by reference, avoiding an argument copy.
      const - prevents modifying the supplied configuration through this reference.

    The implementation stores its own copy in config_ and initialises
    the random generator and Gaussian distribution.

    explicit - prevents certain implicit conversions into PhysicsEngine objects.
    The semicolon means the implementation is provided in PhysicsEngine.cpp.

    */
    explicit PhysicsEngine(const SimulationConfig& config);

    /*
    Calculates the hydrodynamic mobility matrix H using current bead positions.
    H converts applied forces into bead velocities: velocity = H * force.

    const State& state: reads the bead state without copying or modifying it.
    MatrixXd& tensor:  fills the caller's matrix with the calculated mobility.
    void:              no direct return value; tensor receives the result.
    Final const:       does not modify the engine's ordinary stored members.

    */
    void computeHydrodynamicTensor(const State& state, MatrixXd& tensor) const;

    /*
    Computes a lower-triangular Cholesky factor L of the supplied matrix:
        tensor = L * L.transpose()

    Simulation supplies the thermal diffusion matrix D = k_B * T * H.
    The output argument named diffusion therefore receives L, not D itself.

    This factor converts independent Gaussian samples into correlated
    thermal displacements:
        thermal displacement = sqrt(2 * dt) * L * g

    const MatrixXd& tensor: reads the input matrix.
    MatrixXd& diffusion:    writes the resulting factor into this matrix.

    Python equivalent:
        diffusion[:] = np.linalg.cholesky(tensor)
    */
    void computeCholesky(const MatrixXd& tensor, MatrixXd& diffusion) const;

    /*
    Calculates the trap forces and adds the external signal to selected beads.

    state:  supplies bead positions, reference positions, and trap offsets.
    time:   determines the current value of the time-dependent signal.
    forces: receives all force components, ordered as
            [Fx_0, Fy_0, Fx_1, Fy_1, ...].

    The engine reads trap parameters and signal settings from config_.
    It calculates forces; it does not move the beads or switch the traps.


    */
    void computeForces(const State& state, double time, VectorXd& forces) const;

    /*
    Returns one Gaussian random sample with mean 0 and variance 1,
    as configured by the constructor in PhysicsEngine.cpp.

    Simulation later scales and correlates these samples to generate
    thermal displacement.

    No final const: drawing a sample advances the random generator's state.

    */
    double getNormalRandom();

private:
    // These members are internal to PhysicsEngine.
    // Other code uses the public methods to interact with the engine.

    // Retained copy of the physical parameters, including viscosity,
    // bead radius, trap properties, and signal settings.
    SimulationConfig config_;

    // Pseudorandom-number generator: maintains and advances the underlying
    // random sequence. Uses the 64-bit Mersenne Twister algorithm.
    std::mt19937_64 rng_;

    // Gaussian distribution: transforms generator output into normally
    // distributed double-precision samples. Called as dist_(rng_).
    std::normal_distribution<double> dist_;
}; // End of class definition; C++ requires this semicolon.

} // End of namespace modern.
