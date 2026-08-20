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

struct RadarCluster {
    float range_m = 0.f;
    float azimuth_rad = 0.f;
    float range_rate = 0.f;
    int representative_i = -1;
};

float polar_vel_dist(const RadarPoint& a, const RadarPoint& b, float vel_scale)
{
    const float dr = std::abs(a.range_m - b.range_m);
    const float r_avg = 0.5f * (a.range_m + b.range_m);
    const float daz = std::atan2(std::sin(a.azimuth_rad - b.azimuth_rad),
                                 std::cos(a.azimuth_rad - b.azimuth_rad));
    const float dlat = r_avg * std::abs(std::sin(daz));
    const float dv = std::abs(a.range_rate - b.range_rate);
    const float xr = dr / 4.f;
    const float yl = dlat / 0.5f;
    const float zv = dv / vel_scale;
    return std::sqrt(xr * xr + yl * yl + zv * zv);
}

std::vector<RadarCluster> cluster_radar(const std::vector<RadarPoint>& pts, float max_r,
                                        float vel_scale)
{
    const int n = static_cast<int>(pts.size());
    std::vector<int> labels(static_cast<std::size_t>(n), -3);
    int cluster_id = 0;

    auto region = [&](int i) {
        std::vector<int> neighbors;
        if (pts[static_cast<std::size_t>(i)].range_m <= 0.f ||
            pts[static_cast<std::size_t>(i)].range_m > max_r)
            return neighbors;
        for (int j = 0; j < n; ++j) {
            if (pts[static_cast<std::size_t>(j)].range_m <= 0.f ||
                pts[static_cast<std::size_t>(j)].range_m > max_r)
                continue;
            if (polar_vel_dist(pts[static_cast<std::size_t>(i)],
                               pts[static_cast<std::size_t>(j)], vel_scale) <= 1.f)
                neighbors.push_back(j);
        }
        return neighbors;
    };

    for (int i = 0; i < n; ++i) {
        if (labels[static_cast<std::size_t>(i)] != -3) continue;
        const auto neighbors = region(i);
        if (neighbors.empty()) {
            labels[static_cast<std::size_t>(i)] = -2;
            continue;
        }
        if (neighbors.size() < 2) {
            labels[static_cast<std::size_t>(i)] = -1;
            continue;
        }

        labels[static_cast<std::size_t>(i)] = cluster_id;
        std::vector<int> seeds = neighbors;
        for (std::size_t s = 0; s < seeds.size(); ++s) {
            const int j = seeds[s];
            int& label = labels[static_cast<std::size_t>(j)];
            if (label == -1) label = cluster_id;
            if (label != -3) continue;
            label = cluster_id;
            const auto expanded = region(j);
            if (expanded.size() >= 2)
                seeds.insert(seeds.end(), expanded.begin(), expanded.end());
        }
        ++cluster_id;
    }

    std::vector<RadarCluster> clusters;
    clusters.reserve(static_cast<std::size_t>(cluster_id) + pts.size());
    for (int id = 0; id < cluster_id; ++id) {
        float range_sum = 0.f;
        float rate_sum = 0.f;
        float sin_sum = 0.f;
        float cos_sum = 0.f;
        int count = 0;
        int representative = -1;
        for (int i = 0; i < n; ++i) {
            if (labels[static_cast<std::size_t>(i)] != id) continue;
            const auto& p = pts[static_cast<std::size_t>(i)];
            range_sum += p.range_m;
            rate_sum += p.range_rate;
            sin_sum += std::sin(p.azimuth_rad);
            cos_sum += std::cos(p.azimuth_rad);
            if (representative < 0) representative = i;
            ++count;
        }
        if (count > 0) {
            const float inv_n = 1.f / static_cast<float>(count);
            clusters.push_back({range_sum * inv_n, std::atan2(sin_sum, cos_sum),
                                rate_sum * inv_n, representative});
        }
    }

    // Match ZOD: a moving DBSCAN noise point remains a valid one-point cluster.
    for (int i = 0; i < n; ++i) {
        const auto& p = pts[static_cast<std::size_t>(i)];
        if (labels[static_cast<std::size_t>(i)] == -1 &&
            p.range_m > 0.f && p.range_m <= max_r && std::abs(p.range_rate) > 0.5f)
            clusters.push_back({p.range_m, p.azimuth_rad, p.range_rate, i});
    }
    return clusters;
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

int nearest_on_ray(const std::vector<RadarCluster>& clusters, float az_rad,
                   float lat_buf)
{
    int best = -1;
    float best_r = std::numeric_limits<float>::max();
    for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
        const auto& c = clusters[static_cast<std::size_t>(i)];
        const float daz = std::atan2(std::sin(c.azimuth_rad - az_rad),
                                     std::cos(c.azimuth_rad - az_rad));
        if (c.range_m * std::abs(std::sin(daz)) > lat_buf) continue;
        if (c.range_m < best_r) { best_r = c.range_m; best = i; }
    }
    return best;
}

int nearest_on_path(const std::vector<RadarCluster>& clusters, const PathPoly& path,
                    float path_buf)
{
    if (!path.valid) return -1;
    int best = -1;
    float best_dev = std::numeric_limits<float>::max();
    float best_r = std::numeric_limits<float>::max();
    for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
        const auto& c = clusters[static_cast<std::size_t>(i)];
        const float x = c.range_m * std::cos(c.azimuth_rad);
        const float y = c.range_m * std::sin(c.azimuth_rad);
        if (x < 0.5f || x > path.x_max_m) continue;
        const float dev = std::abs(y - path_y(path, x));
        if (dev > path_buf) continue;
        if (dev < best_dev || (dev == best_dev && c.range_m < best_r)) {
            best_dev = dev;
            best_r = c.range_m;
            best = i;
        }
    }
    return best;
}

bool cluster_on_path(const RadarCluster& c, const PathPoly& path, float path_buf)
{
    if (!path.valid) return false;
    const float x = c.range_m * std::cos(c.azimuth_rad);
    const float y = c.range_m * std::sin(c.azimuth_rad);
    if (x < 0.5f || x > path.x_max_m) return false;
    return std::abs(y - path_y(path, x)) <= path_buf;
}

// Labelling Scenario 3: raw points on the path, greedy-group by range+rate, min 2 pts.
int path_groups_raw(const std::vector<RadarPoint>& pts, const PathPoly& path,
                    float path_buf, float max_r, const float* ego_speed_ms,
                    RadarCluster& out)
{
    if (!path.valid) return -1;
    struct Hit { int i; float rg, az, rr, dlat; };
    std::vector<Hit> on_path;
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        const auto& p = pts[static_cast<std::size_t>(i)];
        if (p.range_m <= 0.f || p.range_m > max_r) continue;
        const float x = p.range_m * std::cos(p.azimuth_rad);
        const float y = p.range_m * std::sin(p.azimuth_rad);
        if (x < 0.5f || x > path.x_max_m) continue;
        const float dlat = std::abs(y - path_y(path, x));
        if (dlat > path_buf) continue;
        if (ego_speed_ms) {
            if (std::abs(p.range_rate + *ego_speed_ms) < 0.5f) continue;
        } else if (std::abs(p.range_rate) < 0.5f) {
            continue;
        }
        on_path.push_back({i, p.range_m, p.azimuth_rad, p.range_rate, dlat});
    }
    if (on_path.empty()) return -1;
    std::sort(on_path.begin(), on_path.end(),
              [](const Hit& a, const Hit& b) { return a.rg < b.rg; });

    std::vector<std::vector<Hit>> groups = {{on_path[0]}};
    for (std::size_t n = 1; n < on_path.size(); ++n) {
        const auto& last = groups.back().back();
        if (std::abs(on_path[n].rg - last.rg) <= 4.f &&
            std::abs(on_path[n].rr - last.rr) <= 3.f)
            groups.back().push_back(on_path[n]);
        else
            groups.push_back({on_path[n]});
    }

    int best_i = -1;
    float best_dlat = std::numeric_limits<float>::max();
    float best_r = std::numeric_limits<float>::max();
    RadarCluster best;
    for (const auto& g : groups) {
        if (g.size() < 2) continue;
        float rs = 0.f, rrs = 0.f, ss = 0.f, cs = 0.f, dl = 0.f;
        for (const auto& h : g) {
            rs += h.rg; rrs += h.rr; dl += h.dlat;
            ss += std::sin(h.az); cs += std::cos(h.az);
        }
        const float inv = 1.f / static_cast<float>(g.size());
        const float mean_r = rs * inv;
        const float mean_dl = dl * inv;
        if (mean_dl < best_dlat || (mean_dl == best_dlat && mean_r < best_r)) {
            best_dlat = mean_dl;
            best_r = mean_r;
            best = {mean_r, std::atan2(ss, cs), rrs * inv, g[0].i};
            best_i = g[0].i;
        }
    }
    if (best_i < 0) return -1;
    out = best;
    return best_i;
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
    radar_hist_.clear();
}

CIPOFusionEstimate LongitudinalFusion::update(
    const models::AutoDriveOutput& autodrive,
    const models::AutoSpeedOutput& autospeed,
    const cv::Mat& /*preprocessed_frame*/,
    float dt_s,
    const std::vector<RadarPoint>* radar,
    const PathPoly* path,
    const float* ego_speed_ms)
{
    CIPOFusionEstimate est;
    est.radar.enabled = cfg_.radar_enabled;
    if (radar) est.radar.points = *radar;
    if (path)  est.radar.path   = *path;

    Meas as_h;
    if (autospeed.valid) {
        const auto sel_h = select_cipo(autospeed.detections);
        as_h = sel_h.meas;
        if (!cfg_.radar_enabled)
            est.cut_in_detected = sel_h.cut_in;
        if (as_h.valid) {
            est.as_h_valid  = true;
            est.as_h_dist_m = as_h.distance_m;
        }
    }

    Meas radar_meas;
    if (cfg_.radar_enabled && radar) {
        if (autospeed.valid) {
            const auto sel = select_cipo_radar(autospeed.detections, *radar, path,
                                               dt_s, ego_speed_ms);
            radar_meas          = sel.meas;
            est.cut_in_detected = sel.cut_in;
            est.radar.fov_valid = sel.fov_valid;
            est.radar.fov_az_rad = sel.fov_az_rad;
            est.radar.match_i   = sel.match_i;
            est.radar.hit       = sel.hit;
            est.radar_scenario  = sel.scenario;
        } else if (path) {
            RadarCluster hit;
            if (path_groups_raw(*radar, *path, cfg_.radar_path_buffer_m,
                                cfg_.radar_max_range_m, ego_speed_ms, hit) >= 0) {
                radar_meas.distance_m   = hit.range_m;
                radar_meas.velocity_ms  = hit.range_rate;
                radar_meas.stddev_m     = cfg_.cipo_noise_m;
                radar_meas.stddev_v     = cfg_.cipo_vel_noise_ms;
                radar_meas.valid        = true;
                radar_meas.has_velocity = true;
                est.radar.hit        = RadarHit::Path;
                est.radar.match_i    = hit.representative_i;
                est.radar_scenario   = 3;
            }
        }
        if (radar_meas.valid) {
            est.radar_meas_valid = true;
            est.radar_dist_m     = radar_meas.distance_m;
            est.radar_vel_ms     = radar_meas.velocity_ms;
            est.cipo_raw_found   = true;
            est.cipo_raw_dist_m  = radar_meas.distance_m;
            RadarHist h;
            h.ok = true;
            h.range_m = radar_meas.distance_m;
            h.range_rate = radar_meas.velocity_ms;
            if (est.radar.fov_valid)
                h.az_rad = est.radar.fov_az_rad;
            else if (est.radar.match_i >= 0 &&
                     est.radar.match_i < static_cast<int>(radar->size()))
                h.az_rad = (*radar)[static_cast<std::size_t>(est.radar.match_i)].azimuth_rad;
            radar_hist_.push_back(h);
            if (radar_hist_.size() > 10) radar_hist_.erase(radar_hist_.begin());
        } else {
            radar_hist_.push_back({});
            if (radar_hist_.size() > 10) radar_hist_.erase(radar_hist_.begin());
        }
    } else if (as_h.valid) {
        est.cipo_raw_found  = true;
        est.cipo_raw_dist_m = as_h.distance_m;
    }

    const Meas& cipo_raw = cfg_.radar_enabled ? radar_meas : as_h;

    // ── Step 3: AutoDrive distance (gated by CIPO probability) ───────────────
    static constexpr float D_MAX         = 150.f;
    static constexpr float CIPO_PROB_MIN = 0.40f;  // below this → AD doesn't confirm CIPO

    const bool ad_cipo_confirmed = autodrive.valid &&
                                   autodrive.flag_prob >= CIPO_PROB_MIN;
    const bool autospeed_cipo_confirmed = cipo_raw.valid || as_h.valid;

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
        est.ad_meas_valid  = true;
        est.ad_dist_m      = ad_meas.distance_m;
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
        } else if (as_h.valid) {
            init_from(as_h.distance_m, cfg_.cipo_noise_m);
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

    weight_update(ad_meas, as_h, radar_meas);
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
        char ad_buf[48], ash_buf[32], cr_buf[48];
        if (ad_meas.valid)
            std::snprintf(ad_buf, sizeof(ad_buf), "%.1f m (p=%.0f%% σ=%.1fm)",
                          ad_meas.distance_m,
                          autodrive.valid ? autodrive.flag_prob * 100.f : 0.f,
                          ad_meas.stddev_m);
        else
            std::snprintf(ad_buf, sizeof(ad_buf), "(p=%.0f%% < %.0f%%)",
                          autodrive.valid ? autodrive.flag_prob * 100.f : 0.f,
                          CIPO_PROB_MIN * 100.f);
        if (as_h.valid)
            std::snprintf(ash_buf, sizeof(ash_buf), "%.1f m", as_h.distance_m);
        else
            std::snprintf(ash_buf, sizeof(ash_buf), "(none)");
        if (cipo_raw.valid)
            std::snprintf(cr_buf, sizeof(cr_buf), "%.1f m v=%.2f S%d",
                          cipo_raw.distance_m, cipo_raw.velocity_ms, est.radar_scenario);
        else
            std::snprintf(cr_buf, sizeof(cr_buf), "(none)");

        VP_INFO("[Fusion] AD=%s | AS+H=%s | Radar=%s%s | Fused=%.1f m  v=%.2f m/s  ±%.1f m",
                ad_buf, ash_buf, cr_buf,
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
                                      const PathPoly* path,
                                      float dt_s,
                                      const float* ego_speed_ms) const
{
    auto fill_c = [&](const RadarCluster& c, bool cut_in, RadarHit hit,
                      int scenario, bool fov_ok, float fov_az) {
        CIPOSelection sel;
        sel.meas.distance_m   = c.range_m;
        sel.meas.velocity_ms  = c.range_rate;
        sel.meas.stddev_m     = cfg_.cipo_noise_m;
        sel.meas.stddev_v     = cfg_.cipo_vel_noise_ms;
        sel.meas.valid        = true;
        sel.meas.has_velocity = true;
        sel.cut_in            = cut_in;
        sel.from_path         = hit == RadarHit::Path;
        sel.fov_valid         = fov_ok;
        sel.fov_az_rad        = fov_az;
        sel.match_i           = c.representative_i;
        sel.scenario          = scenario;
        sel.hit               = hit;
        return sel;
    };

    std::vector<models::Detection> cipo_boxes;
    for (const auto& d : dets)
        if (d.class_id == 1 || d.class_id == 2) cipo_boxes.push_back(d);

    // Scenario 1: L1/L2 present — closest box by image-bottom, then FOV cluster.
    if (!cipo_boxes.empty()) {
        std::sort(cipo_boxes.begin(), cipo_boxes.end(),
                  [](const models::Detection& a, const models::Detection& b) {
                      return (a.y1 + a.y2) > (b.y1 + b.y2);
                  });
        const auto& box = cipo_boxes.front();
        const float az = bbox_u_to_radar_az((box.x1 + box.x2) * 0.5f, cfg_);
        const auto clusters = cluster_radar(radar, cfg_.radar_max_range_m, 1.0f);
        int i = nearest_on_ray(clusters, az, 0.5f);
        if (i < 0) i = nearest_on_ray(clusters, az, 1.0f);
        if (i >= 0)
            return fill_c(clusters[static_cast<std::size_t>(i)],
                          box.class_id == 2, RadarHit::Fov, 1, true, az);

        const float dt = (dt_s > 1e-6f) ? dt_s : cfg_.dt_s;
        int best_k = 99;
        RadarCluster from_prev;
        bool have_prev = false;
        for (int k = 1; k <= static_cast<int>(radar_hist_.size()) && k <= 10; ++k) {
            const auto& rj = radar_hist_[radar_hist_.size() - static_cast<std::size_t>(k)];
            if (!rj.ok) continue;
            const float gap = dt * static_cast<float>(k);
            if (gap <= 0.f || gap > 1.f) continue;
            const float daz = std::atan2(std::sin(az - rj.az_rad), std::cos(az - rj.az_rad));
            if (std::abs(daz) * (180.f / kPi) > 4.f) continue;
            const float D_est = rj.range_m + rj.range_rate * gap;
            if (D_est <= 0.f) continue;
            if (k < best_k) {
                best_k = k;
                from_prev = {D_est, az, rj.range_rate, -1};
                have_prev = true;
            }
        }
        if (have_prev)
            return fill_c(from_prev, box.class_id == 2, RadarHit::Fov, 1, true, az);

        if (path) {
            const int ip = nearest_on_path(clusters, *path, 0.5f);
            if (ip >= 0)
                return fill_c(clusters[static_cast<std::size_t>(ip)],
                              box.class_id == 2, RadarHit::Path, 1, true, az);
        }

        CIPOSelection miss;
        miss.fov_valid  = true;
        miss.fov_az_rad = az;
        return miss;
    }

    // Scenario 2: no L1/L2 — other boxes (L3) whose radar cluster sits on the path.
    if (!dets.empty() && path && path->valid) {
        std::vector<models::Detection> others = dets;
        std::sort(others.begin(), others.end(),
                  [](const models::Detection& a, const models::Detection& b) {
                      return (a.y1 + a.y2) > (b.y1 + b.y2);
                  });
        const auto clusters = cluster_radar(radar, cfg_.radar_max_range_m, 1.5f);
        for (const auto& d : others) {
            const float az = bbox_u_to_radar_az((d.x1 + d.x2) * 0.5f, cfg_);
            const int i = nearest_on_ray(clusters, az, 0.5f);
            if (i < 0) continue;
            const auto& c = clusters[static_cast<std::size_t>(i)];
            if (!cluster_on_path(c, *path, 1.0f)) continue;
            return fill_c(c, true, RadarHit::L3, 2, true, az);
        }
    }

    // Scenario 3: no box/radar overlap — moving raw points on the fused path.
    if (path && path->valid) {
        RadarCluster hit;
        if (path_groups_raw(radar, *path, 1.0f, cfg_.radar_max_range_m,
                            ego_speed_ms, hit) >= 0)
            return fill_c(hit, false, RadarHit::Path, 3, false, 0.f);
    }

    return {};
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

void LongitudinalFusion::weight_update(const Meas& ad, const Meas& as_h, const Meas& radar)
{
    for (auto& p : particles_) {
        if (ad.valid)    p.log_w += gaussian_loglik(ad.distance_m,    p.distance_m, ad.stddev_m);
        if (as_h.valid)  p.log_w += gaussian_loglik(as_h.distance_m,  p.distance_m, as_h.stddev_m);
        if (radar.valid) p.log_w += gaussian_loglik(radar.distance_m, p.distance_m, radar.stddev_m);
        if (radar.valid && radar.has_velocity)
            p.log_w += gaussian_loglik(radar.velocity_ms, p.velocity_ms, radar.stddev_v);
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
