#include <planning/longitudinal_planning.hpp>
#include <cmath>
#include <algorithm>

// double dt = 0.05;
double dt = 0.1;

LongitudinalPlanner::LongitudinalPlanner(const Config &config)
    : config_(config) {
}

double LongitudinalPlanner::compute_acceleration(double kappa, double ego_v, bool has_cipo,
                                                 double cipo_v, double cipo_distance) {
    double delta_v = - cipo_v; // CIPO relative speed

    // κ filtering
    kappa_filter_ += config_.kappa_alpha * (std::abs(kappa) - kappa_filter_);

    // curve anticipation
    // Extrapolate t_preview seconds ahead
    double dk = (kappa_filter_ - kappa_prevew_) / dt;
    kappa_prevew_ = kappa_filter_;
    dk_filter_ += config_.dk_alpha * (dk - dk_filter_);

    double kappa_used = kappa_filter_ + config_.t_preview * std::max(0.0, dk_filter_);

    // curve speed limit
    double curv_v_max = (kappa_used > config_.kappa_deadband)
                            ? std::sqrt(config_.a_lat_max / kappa_used)
                            : config_.speed_limit;
    double speed_limit = std::min(config_.speed_limit, curv_v_max);

    double dynamic_term = (ego_v * delta_v) / (2.0 * std::sqrt(config_.a * config_.b));
    double s_star = config_.s0 + std::max(0.0, ego_v * config_.T + dynamic_term);
    double s = std::max(0.5, cipo_distance);

    // IDM
    double free_road_term = std::pow(ego_v / speed_limit, config_.delta);
    double interaction_term = has_cipo ? std::pow(s_star / s, 2.0) : 0.0;

    double accel = config_.a * (1.0 - free_road_term - interaction_term);

    // overspeed branch
    if (ego_v > speed_limit) {
        double brake = -config_.k_over * (ego_v - speed_limit) - config_.a * interaction_term;
        accel = std::min(accel, brake);
    }

    accel = std::clamp(accel, -config_.b, config_.a);

    // jerk limiting
    bool braking_harder = accel < accel_prev_;
    double slew = (braking_harder ? config_.jerk_max_brake : config_.jerk_max) * dt;
    accel = std::clamp(accel, accel_prev_ - slew, accel_prev_ + slew);
    accel_prev_ = accel;

    return accel;
}
