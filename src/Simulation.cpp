#include "Simulation.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <map>

namespace modern {

static constexpr double KB = 1.3806504E-5;

Simulation::Simulation(const std::string& input_file, GeometryType geometry, MobilityType mobility)
    : input_filename_(input_file), state_(0), geometry_arg_(geometry) {
    config_.mobility = mobility;
    parseInput(input_file);
    initializeState();
    engine_ = std::make_unique<PhysicsEngine>(config_);

    int size = 2 * config_.num_beads;
    tensor_.resize(size, size);
    diffusion_.resize(size, size);
    forces_.resize(size);
    noise_.resize(size);
}

void Simulation::parseInput(const std::string& input_file) {
    std::ifstream is(input_file);
    if (!is) throw std::runtime_error("Could not open input file: " + input_file);

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    std::vector<double> p;
    double val;
    while (iss >> val) p.push_back(val);

    for (double value : p)
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite first-line value");
    if (p.size() < 22) throw std::runtime_error("Not enough parameters in input file");

    config_.num_beads       = static_cast<int>(p[0]);
    config_.amplitude       = p[1];
    config_.base_distance   = p[2];
    config_.bead_radius     = p[3];
    config_.dt              = p[4];
    config_.total_time      = p[5];
    config_.viscosity       = p[6];
    config_.alpha           = p[7];
    config_.kx              = p[8];
    config_.beta            = p[9];
    config_.ky              = p[10];
    config_.temperature     = p[11];
    config_.sampling_feedback = static_cast<int>(p[12]);
    config_.sampling_write  = static_cast<int>(p[13]);
    config_.epsilon         = p[14];
    config_.velox           = p[15];
    config_.ks              = p[16];
    config_.fl0             = p[17];
    config_.veloy           = p[18];
    config_.f0_signal       = p[19];
    config_.t_signal        = p[20];
    config_.default_lambda  = p[21];
    config_.geometry        = geometry_arg_;

    int N = config_.num_beads;
    if (p[0] != N) throw std::runtime_error("Noninteger bead count");
    if (N < 1) throw std::runtime_error("num_beads must be positive");
    // Labelled extensions override legacy defaults without changing first-line columns.
    std::map<std::string, std::vector<double>> extensions;
    bool seen_mobility = false;
    while (std::getline(is, line)) {
        line = line.substr(0, line.find('#'));
        std::istringstream extra(line);
        std::string key;
        if (!(extra >> key)) continue;
        if (key == "mobility") {
            std::string choice, excess;
            if (seen_mobility || !(extra >> choice) || (extra >> excess))
                throw std::runtime_error("Malformed/duplicate mobility line");
            seen_mobility = true;
            if (choice == "oseen") config_.mobility = MobilityType::OSEEN;
            else if (choice == "rotne_prager") config_.mobility = MobilityType::ROTNE_PRAGER;
            else if (choice == "blake") config_.mobility = MobilityType::BLAKE;
            else throw std::runtime_error("Unknown mobility");
            continue;
        }
        // Repeated drive lines are intentional; all other extensions are unique.
        if (key == "drive") {
            Drive d;
            int count;
            if (!(extra >> d.amplitude >> d.period >> d.phase >> d.start >> d.stop >> d.waveform >> count)
                || !std::isfinite(d.amplitude) || !std::isfinite(d.period)
                || !std::isfinite(d.phase) || !std::isfinite(d.start) || !std::isfinite(d.stop)
                || d.amplitude < 0 || d.period <= 0 || d.start < 0 || d.stop <= d.start
                || d.waveform < 0 || d.waveform > 2 || count < 1 || count > N)
                throw std::runtime_error("Invalid drive header");
            for (int j=0; j<count; ++j) {
                int bead;
                if (!(extra >> bead) || bead < 0 || bead >= N ||
                    std::find(d.beads.begin(), d.beads.end(), bead) != d.beads.end())
                    throw std::runtime_error("Invalid/duplicate drive target");
                d.beads.push_back(bead);
            }
            std::string excess;
            if (extra >> excess) throw std::runtime_error("Extra drive fields");
            config_.drives.push_back(d);
            continue;
        }
        if (extensions.count(key)) throw std::runtime_error("Duplicate extension: " + key);
        std::vector<double> values;
        std::string token;
        while (extra >> token) {
            size_t used = 0;
            double value = std::stod(token, &used);
            if (used != token.size() || !std::isfinite(value))
                throw std::runtime_error("Non-finite/non-numeric extension: " + key);
            values.push_back(value);
        }
        extensions[key] = values;
    }
    int current = 22;
    if (config_.geometry == GeometryType::CHAIN && p.size() < size_t(22 + N - 1))
        throw std::runtime_error("Missing chain distances");

    if (config_.geometry == GeometryType::CHAIN) {
        for (int i = 0; i < N - 1; ++i) config_.inter_bead_distances.push_back(p[current++]);
    }

    for (int i = 0; i < N; ++i) {
        if (current < (int)p.size()) config_.lambdas.push_back(p[current++]);
        else config_.lambdas.push_back(config_.default_lambda);
    }

    if (config_.geometry == GeometryType::ARBITRARY) {
        for (int i = 0; i < N; ++i) {
            if (current + 1 >= static_cast<int>(p.size()))
                throw std::runtime_error("Missing arbitrary reference coordinates");
            double x = p[current++];
            double y = p[current++];
            config_.initial_positions.push_back({x, y});
        }
    }

    int remaining = static_cast<int>(p.size()) - current;

    if (remaining >= N) {
        for (int i = 0; i < N; ++i) config_.kx_per_bead.push_back(p[current++]);
        remaining -= N;
    }

    if (remaining >= N) {
        for (int i = 0; i < N; ++i) config_.ky_per_bead.push_back(p[current++]);
        remaining -= N;
    }

    if (remaining >= 1) {
        int n_signal = static_cast<int>(p[current++]);
        remaining--;
        for (int i = 0; i < n_signal && remaining > 0; ++i, --remaining)
            config_.signal_beads.push_back(static_cast<int>(p[current++]));
    }

    if (config_.kx_per_bead.empty()) config_.kx_per_bead.assign(N, config_.kx);
    if (config_.ky_per_bead.empty()) config_.ky_per_bead.assign(N, config_.ky);
    if (config_.signal_beads.empty()) config_.signal_beads.push_back(0);
    // Resolve every per-bead property once. Explicit arrays win over defaults.
    config_.angles.assign(N, 0.0);
    config_.alpha_per_bead.assign(N, config_.alpha);
    config_.beta_per_bead.assign(N, config_.beta);
    config_.amplitude_per_bead.assign(N, config_.amplitude);
    config_.epsilon_per_bead.assign(N, config_.epsilon);
    std::map<std::string, std::vector<double>*> arrays = {
        {"angles_deg", &config_.angles}, {"alpha_per_bead", &config_.alpha_per_bead},
        {"beta_per_bead", &config_.beta_per_bead}, {"kx_per_bead", &config_.kx_per_bead},
        {"ky_per_bead", &config_.ky_per_bead}, {"amplitude_per_bead", &config_.amplitude_per_bead},
        {"epsilon_per_bead", &config_.epsilon_per_bead}, {"lambdas", &config_.lambdas}};
    std::map<std::string, double*> scalars = {{"wall_height", &config_.wall_height},
        {"shear_x", &config_.shear_x}, {"shear_y", &config_.shear_y}};
    // Explicit coordinates use the legacy chain first line as a container, then
    // replace its geometry. Thus existing positional formats remain untouched.
    if (extensions.count("reference_x") || extensions.count("reference_y")) {
        if (extensions["reference_x"].size() != size_t(N) || extensions["reference_y"].size() != size_t(N))
            throw std::runtime_error("reference_x/reference_y each need N values");
        config_.geometry = GeometryType::ARBITRARY;
        config_.initial_positions.clear();
        for (int i=0; i<N; ++i)
            config_.initial_positions.push_back({extensions["reference_x"][i], extensions["reference_y"][i]});
        extensions.erase("reference_x"); extensions.erase("reference_y");
    }
    for (const std::string key : {"explicit_drives", "direct_traps", "seed"}) {
        if (!extensions.count(key)) continue;
        const auto values = extensions.at(key);
        if (values.size()!=1 || values[0]!=std::floor(values[0]) ||
            (key=="seed" ? (values[0]<-1 || values[0]>4294967295.0) : (values[0]!=0 && values[0]!=1)))
            throw std::runtime_error("Invalid " + key);
        if (key=="seed") config_.seed=static_cast<long long>(values[0]);
        else if (key=="direct_traps") config_.direct_traps=values[0]!=0;
        else config_.explicit_drives=values[0]!=0;
        extensions.erase(key);
    }
    if (!config_.drives.empty() && !config_.explicit_drives)
        throw std::runtime_error("Drive lines require explicit_drives 1");
    for (const auto& item : extensions) {
        if (arrays.count(item.first)) {
            if (item.second.size() != size_t(N)) throw std::runtime_error(item.first + " needs N values");
            *arrays.at(item.first) = item.second;
        } else if (scalars.count(item.first)) {
            if (item.second.size() != 1) throw std::runtime_error(item.first + " needs one value");
            *scalars.at(item.first) = item.second[0];
        } else throw std::runtime_error("Unknown extension: " + item.first);
    }
    for (double& angle : config_.angles) angle *= M_PI / 180.0;
    for (const auto& item : arrays) {
        if (item.first == "angles_deg" || item.first == "lambdas") continue;
        for (double value : *item.second)
            if (!(value > 0)) throw std::runtime_error(item.first + " must be positive");
    }
    for (int i = 0; i < N; ++i) {
        double index = i - (N - 1) / 2.0;
        if (!config_.direct_traps && !(1.0 + config_.lambdas[i]*(index*index - 0.25) > 0))
            throw std::runtime_error("Nonpositive trap profile");
    }
    if (!(config_.bead_radius > 0 && config_.viscosity > 0 && config_.dt > 0 &&
          config_.total_time > 0 && config_.t_signal > 0 && config_.temperature >= 0 &&
          config_.sampling_feedback > 0 && config_.sampling_write > 0))
        throw std::runtime_error("Invalid physical/numerical scalars");
    if (config_.mobility == MobilityType::BLAKE) {
        if (!(config_.wall_height >= 3*config_.bead_radius))
            throw std::runtime_error("Blake requires wall_height >= 3*bead_radius (far-field guard)");
        if (config_.velox != 0 || config_.veloy != 0)
            throw std::runtime_error("Blake: use shear_x/shear_y instead of uniform velox/veloy");
    } else if (config_.shear_x != 0 || config_.shear_y != 0)
        throw std::runtime_error("shear_x/shear_y require Blake mode");

}

void Simulation::initializeState() {
    int N = config_.num_beads;
    state_ = State(N);

    std::mt19937 rng(config_.seed < 0 ? std::random_device{}() : static_cast<unsigned>(config_.seed));
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    if (config_.geometry == GeometryType::CHAIN) {
        double current_x = 0;
        for (int i = 0; i < N; ++i) {
            state_.reference_pos(2 * i)     = current_x;
            state_.reference_pos(2 * i + 1) = 0;

            double noise = dist(rng) * config_.amplitude_per_bead[i];
            state_.positions(2 * i)     = current_x + noise;
            state_.positions(2 * i + 1) = 0;

            double side = (dist(rng) > 0) ? 1.0 : -1.0;
            state_.trap_offsets(2 * i)     = side * (config_.amplitude_per_bead[i] / 2.0 + config_.epsilon_per_bead[i]);
            state_.trap_offsets(2 * i + 1) = 0;

            if (i < N - 1) current_x += config_.inter_bead_distances[i];
        }
    } else if (config_.geometry == GeometryType::ARBITRARY) {
        for (int i = 0; i < N; ++i) {
            Vector2d pos = config_.initial_positions[i];
            state_.reference_pos(2 * i)     = pos.x();
            state_.reference_pos(2 * i + 1) = pos.y();

            double noise = dist(rng) * config_.amplitude_per_bead[i];
            state_.positions(2 * i)     = pos.x() + noise;
            state_.positions(2 * i + 1) = pos.y();

            double side = (dist(rng) > 0) ? 1.0 : -1.0;
            state_.trap_offsets(2 * i)     = side * (config_.amplitude_per_bead[i] / 2.0 + config_.epsilon_per_bead[i]);
            state_.trap_offsets(2 * i + 1) = 0;
        }
    } else if (config_.geometry == GeometryType::RING) {
        double R = config_.base_distance / (2.0 * std::sin(M_PI / N));
        for (int i = 0; i < N; ++i) {
            double angle = 2.0 * M_PI * i / N;
            double rx    = R * std::cos(angle);
            double ry    = R * std::sin(angle);
            state_.reference_pos(2 * i)     = rx;
            state_.reference_pos(2 * i + 1) = ry;

            double noise = dist(rng) * config_.amplitude_per_bead[i];
            state_.positions(2 * i)     = rx + noise;
            state_.positions(2 * i + 1) = ry;

            double side = (dist(rng) > 0) ? 1.0 : -1.0;
            state_.trap_offsets(2 * i)     = side * (config_.amplitude_per_bead[i] / 2.0 + config_.epsilon_per_bead[i]);
            state_.trap_offsets(2 * i + 1) = 0;
        }
    }
    // Geometry fixes the reference positions; orientation independently fixes
    // the stroke direction. Rotate existing random longitudinal initial states.
    for (int i = 0; i < N; ++i) {
        const Vector2d e(std::cos(config_.angles[i]), std::sin(config_.angles[i]));
        const double q = state_.positions(2*i) - state_.reference_pos(2*i);
        const double offset = state_.trap_offsets(2*i);
        state_.positions.segment<2>(2*i) = state_.reference_pos.segment<2>(2*i) + q * e;
        state_.trap_offsets.segment<2>(2*i) = offset * e;
    }
}

void Simulation::run() {
    std::string title = generateTitle();
    std::filesystem::path p(input_filename_);
    std::string dir_name = p.stem().string();
    if (dir_name.empty()) dir_name = "results";

    std::filesystem::create_directories(dir_name);

    std::ofstream out_dat(dir_name + "/" + title + ".dat");
    std::ofstream out_trap(dir_name + "/" + title + ".trap");
    std::ofstream out_force(dir_name + "/" + title + ".forces");

    // Sidecar travels with the .dat file; old trajectory columns stay unchanged.
    // Python uses these angles for longitudinal phase and spectrum analysis.
    std::ofstream axes(dir_name + "/" + title + ".axes");
    axes << "# angle_degrees reference_x reference_y\n" << std::setprecision(17);
    for (int i = 0; i < config_.num_beads; ++i)
        axes << config_.angles[i] * 180.0 / M_PI << " "
             << state_.reference_pos(2*i) << " " << state_.reference_pos(2*i+1) << "\n";
    axes.close();
    // Separate mobility metadata keeps the established .axes format readable.
    std::ofstream hydro(dir_name + "/" + title + ".hydro");
    hydro << std::setprecision(17) << "mobility "
          << (config_.mobility == MobilityType::BLAKE ? "blake" :
              config_.mobility == MobilityType::OSEEN ? "oseen" : "rotne_prager")
          << "\nwall_height " << config_.wall_height
          << "\nshear_x " << config_.shear_x << "\nshear_y " << config_.shear_y << "\n";
    hydro << "pipeline_version 2\n";
    hydro.close();

    double time    = 0;
    int step_count = 0;
    int frame      = 0;

    std::cout << "Starting Simulation: " << title << " in directory: " << dir_name << std::endl;

    while (time < config_.total_time) {
        if (step_count % config_.sampling_feedback == 0) {
            activeTrapping();
            frame++;
        }

        if (step_count % config_.sampling_write == 0) {
            saveData(out_dat, frame, time);
            saveTraps(out_trap, frame);
            engine_->computeForces(state_, time, forces_);
            saveForces(out_force, frame, time);
        }

        step(time);

        time += config_.dt;
        step_count++;

        if (step_count % 100000 == 0) {
            std::cout << std::fixed << std::setprecision(1)
                      << 100.0 * time / config_.total_time << "% done" << std::endl;
        }
    }
}

void Simulation::step(double time) {
    engine_->computeHydrodynamicTensor(state_, tensor_);
    engine_->computeForces(state_, time, forces_);

    if (config_.temperature > 0) {
        MatrixXd diff_tensor = tensor_ * (KB * config_.temperature);
        engine_->computeCholesky(diff_tensor, diffusion_);

        for (int i = 0; i < noise_.size(); ++i) {
            noise_(i) = engine_->getNormalRandom() * std::sqrt(2.0 * config_.dt);
        }
        noise_ = diffusion_ * noise_;
    } else {
        noise_.setZero();
    }

    VectorXd flow(2 * config_.num_beads);
    for (int i = 0; i < config_.num_beads; ++i) {
        flow(2 * i) = config_.velox + (config_.mobility == MobilityType::BLAKE ? config_.wall_height*config_.shear_x : 0);
        flow(2 * i + 1) = config_.veloy + (config_.mobility == MobilityType::BLAKE ? config_.wall_height*config_.shear_y : 0);
    }

    // Itô drift for the configuration-dependent constrained wall mobility.
    // Deterministic centred differences avoid adding stochastic estimator noise.
    if (config_.temperature > 0 && config_.mobility == MobilityType::BLAKE)
        flow += KB * config_.temperature * engine_->mobilityDivergence(state_);
    state_.positions += (flow * config_.dt) + (tensor_ * forces_ * config_.dt) + noise_;
}

void Simulation::activeTrapping() {
    int N = config_.num_beads;
    for (int i = 0; i < N; ++i) {
        // Switch on displacement ALONG the local stroke, not laboratory x.
        const Vector2d e(std::cos(config_.angles[i]), std::sin(config_.angles[i]));
        double dx = (state_.positions.segment<2>(2*i)
                     - state_.reference_pos.segment<2>(2*i)).dot(e);

        if (dx > (config_.amplitude_per_bead[i] / 2.0)) {
            state_.trap_offsets.segment<2>(2*i) = -(config_.amplitude_per_bead[i] / 2.0 + config_.epsilon_per_bead[i]) * e;
        } else if (dx < -(config_.amplitude_per_bead[i] / 2.0)) {
            state_.trap_offsets.segment<2>(2*i) = (config_.amplitude_per_bead[i] / 2.0 + config_.epsilon_per_bead[i]) * e;
        }
    }
}

void Simulation::saveData(std::ostream& out, int frame, double time) {
    // Record the actual simulation time, not a time inferred from frame.
    out.precision(9);
    out << frame << " " << time << " ";

    out.precision(6);
    for (int i = 0; i < state_.positions.size(); ++i) {
        out << state_.positions(i) << " ";
    }
    out << "\n";
}

void Simulation::saveTraps(std::ostream& out, int frame) {
    out << frame << " ";
    for (int i = 0; i < state_.trap_offsets.size(); ++i) {
        out << state_.reference_pos(i) + state_.trap_offsets(i) << " ";
    }
    out << "\n";
}

void Simulation::saveForces(std::ostream& out, int frame, double time) {
    // Forces must be calculated for this time before calling this function.
    out.precision(9);
    out << frame << " " << time << " ";

    out.precision(6);
    for (int i = 0; i < forces_.size(); ++i) {
        out << forces_(i) << " ";
    }
    out << "\n";
}

std::string Simulation::generateTitle() const {
    std::filesystem::path p(input_filename_);
    return p.stem().string();
}

} // namespace modern
