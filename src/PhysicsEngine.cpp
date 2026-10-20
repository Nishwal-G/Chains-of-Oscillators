#include "PhysicsEngine.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace modern {

PhysicsEngine::PhysicsEngine(const SimulationConfig& config)
    : config_(config), rng_(config.seed < 0 ? std::random_device{}() : static_cast<unsigned>(config.seed)), dist_(0.0, 1.0) {}

void PhysicsEngine::computeHydrodynamicTensor(const State& state, MatrixXd& tensor) const {
    if (config_.mobility == MobilityType::BLAKE) {
        tensor = wallMobility(state);
        return;
    }
    int N    = config_.num_beads;
    double a = config_.bead_radius;
    double zeta     = 6.0 * M_PI * config_.viscosity * a;
    double inv_zeta = 1.0 / zeta;

    tensor.setZero();

    for (int p = 0; p < N; ++p) {
        for (int q = 0; q < N; ++q) {
            if (p == q) {
                tensor.block<2, 2>(2 * p, 2 * q) = Matrix2d::Identity() * inv_zeta;
            } else {
                Vector2d dp = state.positions.segment<2>(2 * q) - state.positions.segment<2>(2 * p);
                double r2   = dp.squaredNorm();
                double r    = std::sqrt(r2);
                if (!(r > 0)) throw std::runtime_error("Coincident beads");

                Matrix2d I       = Matrix2d::Identity();
                Matrix2d unit_rr = (dp * dp.transpose()) / r2;
                Matrix2d D;

                if (config_.mobility == MobilityType::ROTNE_PRAGER) {
                    if (r >= 2 * a) {
                        D = (I + unit_rr + (I - 3.0 * unit_rr) * (2.0 * a * a / (3.0 * r2))) * (3.0 * a / (4.0 * zeta * r));
                    } else {
                        D = (I * (1.0 - 9.0 * r / (32.0 * a)) + unit_rr * (3.0 * r / (32.0 * a))) * inv_zeta;
                    }
                } else {
                    D = (I + unit_rr) * (1.0 / (8.0 * M_PI * config_.viscosity * r));
                }
                tensor.block<2, 2>(2 * p, 2 * q) = D;
            }
        }
    }
}

void PhysicsEngine::computeCholesky(const MatrixXd& tensor, MatrixXd& diffusion) const {
    Eigen::LLT<MatrixXd> llt(tensor);
    if (!tensor.allFinite() || llt.info() != Eigen::Success)
        throw std::runtime_error("Mobility is not positive definite; check separation, height and timestep");
    diffusion = llt.matrixL();
}

void PhysicsEngine::computeForces(const State& state, double time, VectorXd& forces) const {
    int N = config_.num_beads;
    forces.setZero();

    double F0      = config_.f0_signal;
    double Tsignal = config_.t_signal;
    double signal  = (time > 5.0) ? F0 * (std::cos(2.0 * M_PI * time / Tsignal) > 0 ? 1.0 : -1.0) : 0.0;

    double i0 = (N - 1) / 2.0;

    for (int i = 0; i < N; ++i) {
        // Rotate displacement into the bead's own trap frame. kx/alpha now
        // act longitudinally and ky/beta transversely; H is unchanged.
        const double theta = config_.angles[i];
        const Vector2d e(std::cos(theta), std::sin(theta));
        const Vector2d n(-std::sin(theta), std::cos(theta));
        const Vector2d delta = state.positions.segment<2>(2*i)
            - state.reference_pos.segment<2>(2*i) - state.trap_offsets.segment<2>(2*i);
        double dx = delta.dot(e);
        double dy = delta.dot(n);
        double distance_ratio = 1.0;
        if (config_.geometry == GeometryType::CHAIN && N > 1) {
            double current_dist = config_.base_distance;
            if (i == 0)       current_dist = config_.inter_bead_distances[0];
            else if (i == N - 1) current_dist = config_.inter_bead_distances[N - 2];
            else              current_dist = 0.5 * (config_.inter_bead_distances[i-1] + config_.inter_bead_distances[i]);

            distance_ratio = std::pow(current_dist / config_.base_distance, 1.0/3.0);
        }

        double lambda = config_.lambdas[i];
        double C      = 1.0 - 0.25 * lambda;
        double H      = distance_ratio * (lambda * (i - i0) * (i - i0) + C);

        if (config_.direct_traps) H = 1.0; // Explicit physical coefficients, independent of indices.

        double kx_eff = -config_.kx_per_bead[i] * H;
        double ky_eff = -config_.ky_per_bead[i] * H;

        double fx = 0, fy = 0;
        if (std::abs(dx) > 1e-18) fx = kx_eff * dx * config_.alpha_per_bead[i] * std::pow(std::abs(dx), config_.alpha_per_bead[i] - 2);
        if (std::abs(dy) > 1e-18) fy = ky_eff * dy * config_.beta_per_bead[i] * std::pow(std::abs(dy), config_.beta_per_bead[i] - 2);

        if (config_.explicit_drives) {
            for (const Drive& d : config_.drives) {
                if (time < d.start || time >= d.stop ||
                    std::find(d.beads.begin(), d.beads.end(), i) == d.beads.end()) continue;
                const double phase = 2.0*M_PI*time/d.period + d.phase;
                const double wave = d.waveform == 2 ? 1.0 :
                    d.waveform == 1 ? std::sin(phase) : (std::cos(phase)>0 ? 1.0 : -1.0);
                fx += d.amplitude * wave;
            }
        } else if (std::find(config_.signal_beads.begin(), config_.signal_beads.end(), i) != config_.signal_beads.end()) {
            fx += signal; // Exact legacy waveform and t>5 onset remain unchanged.
        }

        // Mobility and noise remain in laboratory coordinates.
        forces.segment<2>(2*i) = fx * e + fy * n;
    }
}

double PhysicsEngine::getNormalRandom() {
    return dist_(rng_);
}

} // namespace modern

namespace modern {
// Blake (1971), as written in Bruot & Cicuta (2016), equations (5)-(6).
// This is the point-force wall Green function, not finite-size RPY-Blake.
Matrix3d PhysicsEngine::blakePair(const Vector3d& target, const Vector3d& source) const {
    const Vector3d r = target - source;
    Vector3d image = source; image.z() = -source.z();
    const Vector3d R = target - image;
    const double d = r.norm(), D = R.norm(), h = source.z();
    if (!(d > 0 && D > 0)) throw std::runtime_error("Singular Blake pair");
    Matrix3d result;
    for (int a=0; a<3; ++a) for (int b=0; b<3; ++b) {
        const double delta = a==b ? 1.0 : 0.0;
        const double sign = b==2 ? -1.0 : 1.0;
        const double stokes = delta/d + r[a]*r[b]/std::pow(d,3);
        const double reflected = delta/D + R[a]*R[b]/std::pow(D,3);
        const double source_doublet = sign*(delta/std::pow(D,3)-3*R[a]*R[b]/std::pow(D,5));
        const double stokes_doublet = sign*(-(a==2 ? R[b] : 0.0)/std::pow(D,3)
            + delta*R.z()/std::pow(D,3) + (b==2 ? R[a] : 0.0)/std::pow(D,3)
            - 3*R[a]*R.z()*R[b]/std::pow(D,5));
        result(a,b) = (stokes-reflected+2*h*h*source_doublet-2*h*stokes_doublet)
                       /(8*M_PI*config_.viscosity);
    }
    return result;
}

MatrixXd PhysicsEngine::wallMobility(const State& state) const {
    const int N = config_.num_beads;
    const double a = config_.bead_radius, h = config_.wall_height;
    const double ratio = a/h, mu = 1/(6*M_PI*config_.viscosity*a);
    // Self terms from the review Eq. (4). Valid away from wall contact;
    // pair terms remain point-force level. Do not use this as lubrication theory.
    const double parallel = mu*(1-9*ratio/16+std::pow(ratio,3)/8-std::pow(ratio,5)/16);
    const double normal = mu*(1-9*ratio/8+std::pow(ratio,3)/2-std::pow(ratio,5)/8);
    MatrixXd full = MatrixXd::Zero(3*N,3*N);
    for(int i=0;i<N;++i) {
        full(3*i,3*i)=parallel; full(3*i+1,3*i+1)=parallel; full(3*i+2,3*i+2)=normal;
        for(int j=i+1;j<N;++j) {
            Vector3d ri(state.positions(2*i),state.positions(2*i+1),h);
            Vector3d rj(state.positions(2*j),state.positions(2*j+1),h);
            // Explicit far-field domain guard, not an overlap regularisation.
            if ((ri-rj).norm() < 4*a)
                throw std::runtime_error("Blake point-force mode requires instantaneous pair distance >= 4 radii");
            Matrix3d block = blakePair(ri,rj);
            full.block<3,3>(3*i,3*j)=block;
            full.block<3,3>(3*j,3*i)=block.transpose();
        }
    }
    // Eliminate the normal constraint forces from vz=0:
    // M_eff = M_pp - M_pz M_zz^{-1} M_zp (Schur complement).
    MatrixXd pp(2*N,2*N), pz(2*N,N), zz(N,N);
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
        pp.block<2,2>(2*i,2*j)=full.block<2,2>(3*i,3*j);
        pz.block<2,1>(2*i,j)=full.block<2,1>(3*i,3*j+2);
        zz(i,j)=full(3*i+2,3*j+2);
    }
    Eigen::LLT<MatrixXd> normal_solver(zz);
    if (!full.allFinite() || normal_solver.info()!=Eigen::Success)
        throw std::runtime_error("Invalid normal wall mobility");
    MatrixXd effective = pp - pz*normal_solver.solve(pz.transpose());
    effective = (0.5*(effective+effective.transpose())).eval();
    Eigen::LLT<MatrixXd> check(effective);
    if (!effective.allFinite() || check.info()!=Eigen::Success)
        throw std::runtime_error("Non-positive constrained wall mobility");
    return effective;
}

VectorXd PhysicsEngine::mobilityDivergence(const State& state) const {
    // Sum_j d M_ij/d q_j in planar coordinates, with fixed height and angles.
    // Cost is 2*(2N) mobility evaluations per thermal step. Check convergence
    // of dt for scientific runs; the numerical derivative uses a relative scale.
    const int size = state.positions.size();
    const double delta = 1e-5*config_.bead_radius;
    VectorXd divergence = VectorXd::Zero(size);
    for(int j=0;j<size;++j) {
        State plus=state, minus=state;
        plus.positions(j)+=delta; minus.positions(j)-=delta;
        divergence += (wallMobility(plus).col(j)-wallMobility(minus).col(j))/(2*delta);
    }
    return divergence;
}
} // namespace modern
