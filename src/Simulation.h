#pragma once

// Provides SimulationConfig, State, GeometryType, and numerical array types.
#include "Types.h"

// Declares the engine used to calculate mobility, forces, and noise factors.
#include "PhysicsEngine.h"

// Provides file streams for reading input and writing results.
#include <fstream>

// Provides std::unique_ptr for managing ownership of the physics engine.
#include <memory>

// Places the class in the modern namespace.
namespace modern {

/*
Simulation controls one experiment: setup, time stepping, trap switching,
and output.

Both class and struct can have public and private members.
The difference is their default access:
    class  -> private by default
    struct -> public by default
*/
class Simulation {
public:
    /*
    Constructor: prepares a simulation using three supplied choices.

    input_file:       filename containing the simulation parameters.
    geometry:         chain, ring, or arbitrary reference geometry.
    use_rotne_prager: true selects Rotne-Prager; false selects Oseen.

    const std::string& receives the filename by reference without modifying it.
    explicit prevents certain implicit conversions into Simulation objects.

    The implementation in Simulation.cpp reads the input, initialises the
    bead state, creates the physics engine, and sizes the working arrays.

    */
    explicit Simulation(const std::string& input_file,
                        GeometryType geometry,
                        bool use_rotne_prager);

    /*
    Runs the prepared experiment.

    Controls the time loop, switching checks, output writes, and calls to step().
    No arguments are needed because the object retains its settings and state.
    void means no value is returned directly; results are written to files.

    */
    void run();

private:
    /*
    Internal operations: outside code cannot call these directly.
    This keeps setup and execution under the control of the constructor and run().
    */

    // Reads input values and stores them in config_.
    // Receives the filename by read-only reference.
    void parseInput(const std::string& input_file);

    // Allocates the bead state and assigns reference coordinates,
    // random initial bead displacements, and initial trap offsets.
    void initializeState();

    // Advances bead positions by one time step using mobility, forces,
    // background flow, and thermal displacement.
    // time supplies the current simulation time for the external signal.
    void step(double time);

    // Checks bead displacements from their reference positions.
    // Updates trap offsets when beads pass the switching thresholds.
    // This changes the active trap; it does not directly move the beads.
    void activeTrapping();

    /*
    Output methods receive:
        out:   a reference to an output stream, usually an open file.
        frame: the feedback counter used by the current output format.

    Passing the stream lets run() open the files while these methods
    handle the formatting.

    */

    // Writes the frame, constructed timestamp, and bead coordinates.
    void saveData(std::ostream& out, int frame);

    // Writes the frame and actual trap-centre coordinates:
    // reference_pos + trap_offsets.
    // Successive records allow us to inspect trap switching.
    void saveTraps(std::ostream& out, int frame);

    // Writes the frame, constructed timestamp, and current force buffer.
    // This method records stored forces; it does not recalculate them.
    void saveForces(std::ostream& out, int frame);

    /*
    Returns the input filename's stem for output naming.
    For example, "templates/6beads.input" produces "6beads".

    std::string means the return value is text.
    Final const means this method does not modify ordinary object members.
    */
    std::string generateTitle() const;

    /*
    Data retained by this simulation object.
    The trailing underscore is a naming convention for class members.

    Python analogues would be attributes such as self.config and self.state.
    */

    // Remembers the input filename so output names can be generated later.
    std::string input_filename_;

    // Stores the experiment's parameters: bead count, time step,
    // fluid properties, trap settings, signal settings, and geometry.
    SimulationConfig config_;

    // Stores actual bead positions, reference positions, and trap offsets.
    // The positions and trap offsets are updated during the run.
    State state_;

    /*
    Exclusively owns the physics-engine object.
    The engine is created after the configuration has been read.
    unique_ptr automatically releases the engine when its ownership ends.

    Accessing its methods uses ->:
        engine_->computeForces(...);

    */
    std::unique_ptr<PhysicsEngine> engine_;

    /*
    Working arrays allocated during setup and reused across time steps.
    With N beads and two coordinates per bead:
        matrices have shape (2N, 2N)
        vectors have length 2N
    */

    // Hydrodynamic mobility matrix H.
    // Converts applied forces into velocities through H * F.
    MatrixXd tensor_;

    // Lower-triangular Cholesky factor L, satisfying:
    // L * L.transpose() = k_B * T * H.
    // Despite its name, this stores the factor, not the full diffusion matrix.
    MatrixXd diffusion_;

    // Applied forces, ordered [Fx_0, Fy_0, Fx_1, Fy_1, ...].
    // Includes trap forces and the external signal on selected beads.
    VectorXd forces_;

    // Thermal displacement for one time step.
    // After scaling and correlation: sqrt(2 * dt) * L * g.
    VectorXd noise_;

    // Remembers the constructor's geometry choice so parseInput()
    // can place it in config_ before reading geometry-dependent values.
    GeometryType geometry_arg_;

}; // End of class definition; the semicolon is required.

} // End of namespace modern.
