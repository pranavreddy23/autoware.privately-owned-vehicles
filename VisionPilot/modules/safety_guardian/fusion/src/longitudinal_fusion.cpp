#include <fusion/longitudinal_fusion.hpp>
#include <logging/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace visionpilot::fusion {
namespace {

constexpr float kModelW = 1024.f;
constexpr float kPi     = 3.14159265f;

float path_y(const PathPoly& p, float x) { return p.a * x * x + p.b * x + p.c; }

// Same ±1 m world-metre corridor as production path fill. Not pixels.
bool in_path_corridor(const RadarPoint& p, const PathPoly& path, float buf, float max_r)
{
    if (!path.valid || p.range_m <= 0.f || p.range_m > max_r) return false;
    const float x = p.range_m * std::cos(p.azimuth_rad);
    const float y = p.range_m * std::sin(p.azimuth_rad);
    if (x < 0.5f || x > path.x_max_m) return false;
    return std::abs(y - path_y(path, x)) <= buf;
}

float bbox_u_to_radar_az(float u, const LongitudinalFusion::Config& cfg)
{
    const float h_deg = ((u - kModelW * 0.5f) / (kModelW * 0.5f)) * (cfg.radar_hfov_deg * 0.5f);
    const float h = h_deg * (kPi / 180.f);
    const cv::Matx33d R_cam(cfg.cam_T(0,0), cfg.cam_T(0,1), cfg.cam_T(0,2),
                            cfg.cam_T(1,0), cfg.cam_T(1,1), cfg.cam_T(1,2),
                            cfg.cam_T(2,0), cfg.cam_T(2,1), cfg.cam_T(2,2));
    const cv::Matx33d R_radar(cfg.radar_T(0,0), cfg.radar_T(0,1), cfg.radar_T(0,2),
                              cfg.radar_T(1,0), cfg.radar_T(1,1), cfg.radar_T(1,2),
                              cfg.radar_T(2,0), cfg.radar_T(2,1), cfg.radar_T(2,2));
    const cv::Vec3d dir_radar = R_radar.t() * (R_cam * cv::Vec3d(std::sin(h), 0.0, std::cos(h)));
    return static_cast<float>(std::atan2(dir_radar[1], dir_radar[0]));
}

int nearest_on_ray(const std::vector<RadarPoint>& pts, float az_rad,
                   float lat_buf, float max_r,
                   const PathPoly* path, float path_buf)
{
    int best = -1;
    float best_r = max_r;
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        const auto& p = pts[i];
        if (p.range_m <= 0.f || p.range_m > max_r) continue;
        if (path && path->valid && !in_path_corridor(p, *path, path_buf, max_r))
            continue;
        const float daz = std::atan2(std::sin(p.azimuth_rad - az_rad),
                                     std::cos(p.azimuth_rad - az_rad));
        if (p.range_m * std::abs(std::sin(daz)) > lat_buf) continue;
        if (p.range_m < best_r) { best_r = p.range_m; best = i; }
    }
    return best;
}

int nearest_on_path(const std::vector<RadarPoint>& pts, const PathPoly& path,
                    float path_buf, float max_r)
{
    if (!path.valid) return -1;
    int best = -1;
    float best_r = max_r;
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        if (!in_path_corridor(pts[i], path, path_buf, max_r)) continue;
        if (pts[i].range_m < best_r) { best_r = pts[i].range_m; best = i; }
    }
    return best;
}

}  // namespace

// ─── Construction ─────────────────────────────────────────────────────────────

LongitudinalFusion::LongitudinalFusion()
    : LongitudinalFusion(Config{})
{}

LongitudinalFusion::LongitudinalFusion(Config cfg)
    : cfg_(cfg)
    , rng_(std::random_device{}())
{
    if (cfg_.n_particles < 10)
        throw std::invalid_argument("LongitudinalFusion: n_particles must be >= 10");
    particles_.reserve(static_cast<std::size_t>(cfg_.n_particles));
}

// ─── Public API ───────────────────────────────────────────────────────────────

void LongitudinalFusion::reset()
{
    particles_.clear();
    initialised_ = false;
}

CIPOFusionEstimate LongitudinalFusion::update(
    const models::AutoDriveOutput& autodrive,
    const models::AutoSpeedOutput& autospeed,
    const cv::Mat& /*preprocessed_frame*/,
    float dt_s,
    const std::vector<RadarPoint>* radar,
    const PathPoly* path)
{
    CIPOFusionEstimate est;
    est.radar.enabled = cfg_.radar_enabled;
    if (radar) est.radar.points = *radar;
    if (path)  est.radar.path   = *path;

    Meas cipo_raw;
    if (cfg_.radar_enabled && radar) {
        if (autospeed.valid) {
            const auto sel = select_cipo_radar(autospeed.detections, *radar, path);
            cipo_raw            = sel.meas;
            est.cut_in_detected = sel.cut_in;
            est.radar.fov_valid = sel.fov_valid;
            est.radar.fov_az_rad = sel.fov_az_rad;
            est.radar.match_i   = sel.match_i;
            if (sel.meas.valid)
                est.radar.hit = sel.from_path ? RadarHit::Path : RadarHit::Fov;
        } else if (path) {
            const int i = nearest_on_path(*radar, *path, cfg_.radar_path_buffer_m,
                                          cfg_.radar_max_range_m);
            if (i >= 0) {
                const auto& hit = (*radar)[static_cast<std::size_t>(i)];
                cipo_raw.distance_m   = hit.range_m;
                cipo_raw.velocity_ms  = hit.range_rate;
                cipo_raw.stddev_m     = cfg_.cipo_noise_m;
                cipo_raw.stddev_v     = cfg_.cipo_vel_noise_ms;
                cipo_raw.valid        = true;
                cipo_raw.has_velocity = true;
                est.radar.hit     = RadarHit::Path;
                est.radar.match_i = i;
            }
        }
        if (cipo_raw.valid) {
            est.cipo_raw_found  = true;
            est.cipo_raw_dist_m = cipo_raw.distance_m;
        }
    } else if (autospeed.valid) {
        const auto sel = select_cipo(autospeed.detections);
        cipo_raw            = sel.meas;
        est.cut_in_detected = sel.cut_in;
        if (cipo_raw.valid) {
            est.cipo_raw_found  = true;
            est.cipo_raw_dist_m = cipo_raw.distance_m;
        }
    }

    // ── Step 3: AutoDrive distance (gated by CIPO probability) ───────────────
    static constexpr float D_MAX         = 150.f;
    static constexpr float CIPO_PROB_MIN = 0.40f;  // below this → AD doesn't confirm CIPO

    const bool ad_cipo_confirmed = autodrive.valid &&
                                   autodrive.flag_prob >= CIPO_PROB_MIN;
    const bool autospeed_cipo_confirmed = cipo_raw.valid;

    // If neither network confirms a CIPO target, report max distance and
    // reset the particle filter so we start fresh when a target reappears.
    if (!ad_cipo_confirmed && !autospeed_cipo_confirmed) {
        if (initialised_) {
            VP_INFO("[Fusion] No CIPO confirmed (AD=%.0f%%  AS=none) — reset to %.0f m",
                    autodrive.valid ? autodrive.flag_prob * 100.f : 0.f, D_MAX);
            reset();
        }
        est.valid      = true;
        est.distance_m = D_MAX;
        return est;
    }

    Meas ad_meas;
    if (ad_cipo_confirmed) {
        ad_meas.distance_m = D_MAX * (1.f - autodrive.dist_normalized);
        const float p      = std::clamp(autodrive.flag_prob, 0.f, 1.f);
        ad_meas.stddev_m   = cfg_.autodrive_noise_min_m +
                             (cfg_.autodrive_noise_m - cfg_.autodrive_noise_min_m) * (1.f - p);
        ad_meas.valid      = true;
    }

    // ── Step 4: Particle filter ───────────────────────────────────────────────
    const float dt = (dt_s > 1e-6f) ? dt_s : cfg_.dt_s;

    if (!initialised_) {
        if (ad_meas.valid) {
            init_from(ad_meas.distance_m, ad_meas.stddev_m);
        } else if (cipo_raw.valid) {
            init_from(cipo_raw.distance_m, cfg_.cipo_noise_m,
                      cipo_raw.has_velocity ? cipo_raw.velocity_ms : 0.f,
                      cipo_raw.has_velocity ? cipo_raw.stddev_v : 2.f);
        } else {
            return est;
        }
        initialised_ = true;
    } else {
        predict(dt);

        float cloud_mean = 0.f;
        for (const auto& p : particles_) cloud_mean += p.distance_m;
        cloud_mean /= static_cast<float>(particles_.size());

        const float gate = cfg_.reset_gate_m;

        if (cipo_raw.valid &&
            std::abs(cipo_raw.distance_m - cloud_mean) > gate) {
            VP_INFO("[Fusion] Target change — reinit %.1f→%.1f m (CIPO)  gate=%.1f",
                    cloud_mean, cipo_raw.distance_m, gate);
            init_from(cipo_raw.distance_m, cfg_.cipo_noise_m,
                      cipo_raw.has_velocity ? cipo_raw.velocity_ms : 0.f,
                      cipo_raw.has_velocity ? cipo_raw.stddev_v : 2.f);
        } else if (ad_meas.valid &&
                   (ad_meas.distance_m - cloud_mean) > gate) {
            VP_INFO("[Fusion] Cut-out — reinit %.1f→%.1f m (AD)  gate=%.1f",
                    cloud_mean, ad_meas.distance_m, gate);
            init_from(ad_meas.distance_m, ad_meas.stddev_m);
        }
    }

    weight_update(ad_meas, cipo_raw);
    if (effective_n() < 0.5f * static_cast<float>(cfg_.n_particles)) resample();

    // ── Step 5: Posterior mean — weighted particle average (MRPT getMean style) ─
    // Both distance and velocity come directly from the particle ensemble.
    // No EMA, no finite difference, no deadband needed.
    const auto w = linear_weights();
    const auto N = particles_.size();
    float mean_d = 0.f, mean_v = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        mean_d += w[i] * particles_[i].distance_m;
        mean_v += w[i] * particles_[i].velocity_ms;
    }
    float var_d = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        const float dd = particles_[i].distance_m - mean_d;
        var_d += w[i] * dd * dd;
    }

    est.valid             = true;
    est.distance_m        = mean_d;
    est.velocity_ms       = mean_v;
    est.distance_stddev_m = std::sqrt(std::max(0.f, var_d));

    // ── Step 7: Debug log ─────────────────────────────────────────────────────
    if (cfg_.debug) {
        char ad_buf[48], cr_buf[32];
        if (ad_meas.valid)
            std::snprintf(ad_buf, sizeof(ad_buf), "%.1f m (p=%.0f%% σ=%.1fm)",
                          ad_meas.distance_m,
                          autodrive.valid ? autodrive.flag_prob * 100.f : 0.f,
                          ad_meas.stddev_m);
        else
            std::snprintf(ad_buf, sizeof(ad_buf), "(p=%.0f%% < %.0f%%)",
                          autodrive.valid ? autodrive.flag_prob * 100.f : 0.f,
                          CIPO_PROB_MIN * 100.f);
        if (cipo_raw.valid)
            std::snprintf(cr_buf, sizeof(cr_buf), "%.1f m", cipo_raw.distance_m);
        else
            std::snprintf(cr_buf, sizeof(cr_buf), "(none)");

        VP_INFO("[Fusion] AD=%s | CIPO=%s%s | Fused=%.1f m  v=%.2f m/s  ±%.1f m",
                ad_buf, cr_buf,
                est.cut_in_detected ? " [CUT-IN]" : "",
                est.distance_m, est.velocity_ms, est.distance_stddev_m);
    }

    return est;
}

// ─── Homography projection ────────────────────────────────────────────────────

float LongitudinalFusion::project_dist(const cv::Mat& H, float ux, float uy)
{
    std::vector<cv::Point2f> src = {cv::Point2f(ux, uy)}, dst;
    cv::perspectiveTransform(src, dst, H);
    return std::sqrt(dst[0].x * dst[0].x + dst[0].y * dst[0].y);
}

// Find the closest Level 1 and closest Level 2 detection.
// If Level 2 is closer than Level 1 → cut-in: fuse the Level 2 distance.
// Otherwise fuse the closest Level 1.
// class_id == 1: CIPO Level 1 (straight ahead, same lane)
// class_id == 2: CIPO Level 2 (adjacent lane, potential cut-in)
LongitudinalFusion::CIPOSelection
LongitudinalFusion::select_cipo(const std::vector<models::Detection>& dets) const
{
    float best_l1 = std::numeric_limits<float>::max();
    float best_l2 = std::numeric_limits<float>::max();

    for (const auto& d : dets) {
        if (d.class_id != 1 && d.class_id != 2) continue;
        const float cx   = (d.x1 + d.x2) * 0.5f;
        const float dist = project_dist(H_, cx, d.y2);
        if (cfg_.debug) {
            VP_INFO("[CIPO-DBG] cls=%d  bbox=(%.0f,%.0f,%.0f,%.0f)  bottom-center=(%.0f,%.0f) → %.1f m",
                    d.class_id, d.x1, d.y1, d.x2, d.y2, cx, d.y2, dist);
        }
        if (d.class_id == 1 && dist < best_l1) best_l1 = dist;
        if (d.class_id == 2 && dist < best_l2) best_l2 = dist;
    }

    CIPOSelection sel;
    sel.meas.stddev_m = cfg_.cipo_noise_m;

    const bool have_l1 = best_l1 < std::numeric_limits<float>::max();
    const bool have_l2 = best_l2 < std::numeric_limits<float>::max();

    if (have_l2 && have_l1 && best_l2 < best_l1) {
        sel.meas.distance_m = best_l2;
        sel.meas.valid      = true;
        sel.cut_in          = true;
    } else if (have_l1) {
        sel.meas.distance_m = best_l1;
        sel.meas.valid      = true;
    }
    return sel;
}

LongitudinalFusion::CIPOSelection
LongitudinalFusion::select_cipo_radar(const std::vector<models::Detection>& dets,
                                      const std::vector<RadarPoint>& radar,
                                      const PathPoly* path) const
{
    auto fill = [&](int i, bool cut_in, bool from_path, bool fov_ok, float fov_az) {
        CIPOSelection sel;
        const auto& hit = radar[static_cast<std::size_t>(i)];
        sel.meas.distance_m   = hit.range_m;
        sel.meas.velocity_ms  = hit.range_rate;
        sel.meas.stddev_m     = cfg_.cipo_noise_m;
        sel.meas.stddev_v     = cfg_.cipo_vel_noise_ms;
        sel.meas.valid        = true;
        sel.meas.has_velocity = true;
        sel.cut_in            = cut_in;
        sel.from_path         = from_path;
        sel.fov_valid         = fov_ok;
        sel.fov_az_rad        = fov_az;
        sel.match_i           = i;
        return sel;
    };

    int i_l1 = -1, i_l2 = -1;
    float az_l1 = 0.f, az_l2 = 0.f;
    bool have_az_l1 = false, have_az_l2 = false;
    for (const auto& d : dets) {
        if (d.class_id != 1 && d.class_id != 2) continue;
        const float az = bbox_u_to_radar_az((d.x1 + d.x2) * 0.5f, cfg_);
        if (d.class_id == 1 && !have_az_l1) { az_l1 = az; have_az_l1 = true; }
        if (d.class_id == 2 && !have_az_l2) { az_l2 = az; have_az_l2 = true; }
        const int i = nearest_on_ray(radar, az, cfg_.radar_lat_buffer_m,
                                     cfg_.radar_max_range_m, path, cfg_.radar_path_buffer_m);
        if (i < 0) continue;
        if (d.class_id == 1 && (i_l1 < 0 || radar[i].range_m < radar[i_l1].range_m)) {
            i_l1 = i; az_l1 = az; have_az_l1 = true;
        }
        if (d.class_id == 2 && (i_l2 < 0 || radar[i].range_m < radar[i_l2].range_m)) {
            i_l2 = i; az_l2 = az; have_az_l2 = true;
        }
    }
    const bool fov_ok = have_az_l1 || have_az_l2;
    const float fov_fallback = have_az_l1 ? az_l1 : az_l2;
    if (i_l2 >= 0 && i_l1 >= 0 && radar[i_l2].range_m < radar[i_l1].range_m)
        return fill(i_l2, true, false, true, az_l2);
    if (i_l1 >= 0) return fill(i_l1, false, false, true, az_l1);
    if (i_l2 >= 0) return fill(i_l2, true, false, true, az_l2);

    const int i_path = path
        ? nearest_on_path(radar, *path, cfg_.radar_path_buffer_m, cfg_.radar_max_range_m)
        : -1;
    if (i_path >= 0) return fill(i_path, false, true, fov_ok, fov_fallback);

    CIPOSelection miss;
    miss.fov_valid  = fov_ok;
    miss.fov_az_rad = fov_fallback;
    return miss;
}

// ─── Particle filter internals ────────────────────────────────────────────────

void LongitudinalFusion::init_from(float dist_m, float stddev_m, float vel_ms, float vel_std)
{
    particles_.resize(static_cast<std::size_t>(cfg_.n_particles));
    std::normal_distribution<float> nd(dist_m, stddev_m);
    std::normal_distribution<float> nv(vel_ms, vel_std);
    for (auto& p : particles_) {
        p.distance_m  = std::clamp(nd(rng_), 0.f, cfg_.d_max_m);
        p.velocity_ms = nv(rng_);
        p.log_w       = 0.f;
    }
}

void LongitudinalFusion::predict(float dt_s)
{
    std::normal_distribution<float> nd(0.f, cfg_.process_noise_dist_m);
    std::normal_distribution<float> nv(0.f, cfg_.process_noise_vel_ms);
    for (auto& p : particles_) {
        p.distance_m  = std::clamp(p.distance_m + p.velocity_ms * dt_s + nd(rng_), 0.f, cfg_.d_max_m);
        p.velocity_ms = p.velocity_ms + nv(rng_);
    }
}

float LongitudinalFusion::gaussian_loglik(float z, float mean, float sigma)
{
    const float d = z - mean;
    return -0.5f * (d / sigma) * (d / sigma);
}

void LongitudinalFusion::weight_update(const Meas& ad, const Meas& cipo_raw)
{
    for (auto& p : particles_) {
        if (ad.valid)       p.log_w += gaussian_loglik(ad.distance_m,       p.distance_m, ad.stddev_m);
        if (cipo_raw.valid) p.log_w += gaussian_loglik(cipo_raw.distance_m, p.distance_m, cipo_raw.stddev_m);
        if (cipo_raw.valid && cipo_raw.has_velocity)
            p.log_w += gaussian_loglik(cipo_raw.velocity_ms, p.velocity_ms, cipo_raw.stddev_v);
    }
}

std::vector<float> LongitudinalFusion::linear_weights() const
{
    const std::size_t N = particles_.size();
    float max_lw = particles_[0].log_w;
    for (const auto& p : particles_) max_lw = std::max(max_lw, p.log_w);

    std::vector<float> w(N);
    float sum = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        w[i] = std::exp(particles_[i].log_w - max_lw);
        sum  += w[i];
    }
    if (sum < 1e-12f) {
        const float w0 = 1.f / static_cast<float>(N);
        for (auto& wi : w) wi = w0;
    } else {
        for (auto& wi : w) wi /= sum;
    }
    return w;
}

float LongitudinalFusion::effective_n() const
{
    const auto w = linear_weights();
    float ss = 0.f;
    for (auto wi : w) ss += wi * wi;
    return 1.f / (ss + 1e-12f);
}

void LongitudinalFusion::resample()
{
    const int N = static_cast<int>(particles_.size());
    if (N == 0) return;

    const auto w = linear_weights();
    std::vector<float> cs(static_cast<std::size_t>(N));
    cs[0] = w[0];
    for (int i = 1; i < N; ++i)
        cs[static_cast<std::size_t>(i)] =
            cs[static_cast<std::size_t>(i-1)] + w[static_cast<std::size_t>(i)];

    std::vector<Particle> np;
    np.reserve(static_cast<std::size_t>(N));
    std::uniform_real_distribution<float> u(0.f, 1.f / static_cast<float>(N));
    const float u0 = u(rng_);
    int j = 0;
    for (int i = 0; i < N; ++i) {
        const float thr = u0 + static_cast<float>(i) / static_cast<float>(N);
        while (j < N-1 && cs[static_cast<std::size_t>(j)] < thr) ++j;
        np.push_back({particles_[static_cast<std::size_t>(j)].distance_m,
                      particles_[static_cast<std::size_t>(j)].velocity_ms,
                      0.f});
    }
    particles_ = std::move(np);
}

}  // namespace visionpilot::fusion
