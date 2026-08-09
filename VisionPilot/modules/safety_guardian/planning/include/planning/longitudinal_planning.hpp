#ifndef VISIONPILOT_LONGITUDIANL_PLANNING_HPP
#define VISIONPILOT_LONGITUDIANL_PLANNING_HPP

extern double dt; // 50Hz control period — still used by planning.cpp

class LongitudinalPlanner {
public:
    struct Config {
        double speed_limit = 20.0; // default value
        double a = 1.2;
        double b = 3.5;
        double T = 1.5;
        double s0 = 2.0;
        double delta = 4.0;

        // --- curve anticipation ---
        double t_preview = 2.5;
        // a full second further ahead
        double dk_alpha = 0.15;

        // --- curve handling ---
        double a_lat_max = 1.0;
        double kappa_alpha = 0.4;
        double kappa_deadband = 5e-4;
        double k_over = 1.8;

        // --- comfort / smoothness ---
        double jerk_max = 1.5;
        double jerk_max_brake = 3.0;
    };

    explicit LongitudinalPlanner(const Config &config);

    // Returns the ego acceleration value for the current step,
    // jerk-limited and clamped to [-b, a].
    //
    //   kappa         : lane curvature (1/m);
    //   ego_v         : ego speed (m/s)
    //   has_cipo      : CIPO in front
    //   cipo_v        : lead speed RELATIVE to ego (v_lead - v_ego): negative when ego is closing
    //   cipo_distance : bumper-to-bumper gap (m); use 9999.0 for free road
    double compute_acceleration(double kappa, double ego_v, bool has_cipo,
                                double cipo_v, double cipo_distance);

private:
    Config config_;
    double kappa_filter_ = 0.0; // low-passed |κ|
    double kappa_prevew_ = 0.0; // previous filtered |κ| (for the slope)
    double dk_filter_ = 0.0; // low-passed dκ/dt
    double accel_prev_ = 0.0; // last issued command, for jerk limiting
};

#endif // VISIONPILOT_LONGITUDIANL_PLANNING_HPP
