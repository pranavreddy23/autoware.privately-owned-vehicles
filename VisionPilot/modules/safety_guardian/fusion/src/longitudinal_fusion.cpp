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
    // Indices of the returns this cluster is made of. Published to the debug view
    // so the BEV paints the set association actually used, not a second opinion.
    std::vector<int> members;
};

// Neighbourhood of one vehicle, not two. 3 m range / 2 m lateral was wide
// enough to glue a lead car to the van in the next metre of the same beam.
constexpr float kClusterRangeM = 2.0f;
constexpr float kClusterLatM   = 1.5f;
constexpr float kMovingAbsMs   = 0.5f;

float polar_vel_dist(const RadarPoint& a, const RadarPoint& b, float vel_scale)
{
    const float dr = std::abs(a.range_m - b.range_m);
    const float r_avg = 0.5f * (a.range_m + b.range_m);
    const float daz = std::atan2(std::sin(a.azimuth_rad - b.azimuth_rad),
                                 std::cos(a.azimuth_rad - b.azimuth_rad));
    const float dlat = r_avg * std::abs(std::sin(daz));
    const float dv = std::abs(a.range_rate - b.range_rate);
    const float xr = dr / kClusterRangeM;
    const float yl = dlat / kClusterLatM;
    const float zv = dv / vel_scale;
    return std::sqrt(xr * xr + yl * yl + zv * zv);
}

float estimate_ego_ms(const std::vector<RadarPoint>& pts)
{
    std::vector<float> rr;
    rr.reserve(pts.size());
    for (const auto& p : pts) {
        const float ca = std::cos(p.azimuth_rad);
        if (p.range_m > 2.f && ca > 0.2f) rr.push_back(p.range_rate / ca);
    }
    if (rr.size() < 8) return 0.f;
    std::nth_element(rr.begin(), rr.begin() + static_cast<long>(rr.size() / 2), rr.end());
    return -rr[rr.size() / 2];
}

bool cluster_is_moving(const RadarCluster& c, float ego_ms, bool have_ego)
{
    const float abs_v = have_ego
        ? std::abs(c.range_rate + ego_ms * std::cos(c.azimuth_rad))
        : std::abs(c.range_rate);
    return abs_v > kMovingAbsMs;
}

std::vector<RadarCluster> cluster_radar(const std::vector<RadarPoint>& pts, float max_r,
                                        float vel_scale, float ego_ms, bool have_ego)
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
        int representative = -1;
        std::vector<int> members;
        for (int i = 0; i < n; ++i) {
            if (labels[static_cast<std::size_t>(i)] != id) continue;
            const auto& p = pts[static_cast<std::size_t>(i)];
            range_sum += p.range_m;
            rate_sum += p.range_rate;
            sin_sum += std::sin(p.azimuth_rad);
            cos_sum += std::cos(p.azimuth_rad);
            if (representative < 0) representative = i;
            members.push_back(i);
        }
        if (!members.empty()) {
            const float inv_n = 1.f / static_cast<float>(members.size());
            clusters.push_back({range_sum * inv_n, std::atan2(sin_sum, cos_sum),
                                rate_sum * inv_n, representative, std::move(members)});
        }
    }

    // A moving DBSCAN-noise point is still a one-point cluster. Static noise
    // is not — a lone rail return needs a second point before it is a CIPO.
    for (int i = 0; i < n; ++i) {
        const auto& p = pts[static_cast<std::size_t>(i)];
        if (labels[static_cast<std::size_t>(i)] != -1) continue;
        if (p.range_m <= 0.f || p.range_m > max_r) continue;
        const float abs_v = have_ego
            ? std::abs(p.range_rate + ego_ms * std::cos(p.azimuth_rad))
            : std::abs(p.range_rate);
        if (abs_v > kMovingAbsMs)
            clusters.push_back({p.range_m, p.azimuth_rad, p.range_rate, i, {i}});
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

// Model-based clustering window (Li, Stolz, Feng, Kunert — ICVES 2019).
// Density thresholds alone cannot group a sparse co-speed vehicle, and widening
// them to try makes roadside structure condense into phantom in-path targets.
// Instead, group every return inside the object's own predicted contour whatever
// the local density, then region-grow once with the ordinary neighbourhood to
// recover returns just outside it. The paper predicts the contour from a tracked
// radar cluster; a camera box gives us the same contour on the first frame.
int cluster_in_window(const std::vector<RadarPoint>& pts,
                      float az_min, float az_max, float r_min, float r_max,
                      float az_c, float lat_max_m,
                      float vel_scale, RadarCluster& out)
{
    // A bearing window keeps a constant width in angle, so its width in metres
    // grows with range: the same box that spans 1.8 m at 40 m spans 6 m at 130 m,
    // which is a lane and a half and lets the next lane over win. An object is
    // the same width however far away it is, so bound the offset from the centre
    // ray in metres as well.
    const auto in_lat = [&](const RadarPoint& p) {
        const float daz = std::atan2(std::sin(p.azimuth_rad - az_c),
                                     std::cos(p.azimuth_rad - az_c));
        return p.range_m * std::abs(std::sin(daz)) <= lat_max_m;
    };

    // One object has one Doppler. Without this band the region-grow chains
    // through the velocity dimension in small steps and walks off a co-speed
    // vehicle onto the static world, averaging the two into a range-rate that
    // describes neither.
    constexpr float kRateBandMs = 2.f;

    const int n = static_cast<int>(pts.size());
    std::vector<char> member(static_cast<std::size_t>(n), 0);
    std::vector<float> seed_rates;
    for (int i = 0; i < n; ++i) {
        const auto& p = pts[static_cast<std::size_t>(i)];
        if (p.range_m < r_min || p.range_m > r_max) continue;
        if (p.azimuth_rad < az_min || p.azimuth_rad > az_max) continue;
        if (!in_lat(p)) continue;
        member[static_cast<std::size_t>(i)] = 1;
        seed_rates.push_back(p.range_rate);
    }
    if (seed_rates.empty()) return -1;

    std::nth_element(seed_rates.begin(),
                     seed_rates.begin() + static_cast<long>(seed_rates.size() / 2),
                     seed_rates.end());
    const float rate_ref = seed_rates[seed_rates.size() / 2];

    bool any = false;
    for (int i = 0; i < n; ++i) {
        if (!member[static_cast<std::size_t>(i)]) continue;
        if (std::abs(pts[static_cast<std::size_t>(i)].range_rate - rate_ref) > kRateBandMs)
            member[static_cast<std::size_t>(i)] = 0;
        else
            any = true;
    }
    if (!any) return -1;

    const std::vector<char> seeds = member;
    for (int i = 0; i < n; ++i) {
        if (!seeds[static_cast<std::size_t>(i)]) continue;
        for (int j = 0; j < n; ++j) {
            if (member[static_cast<std::size_t>(j)]) continue;
            const auto& q = pts[static_cast<std::size_t>(j)];
            if (q.range_m <= 0.f) continue;
            if (!in_lat(q)) continue;
            if (std::abs(q.range_rate - rate_ref) > kRateBandMs) continue;
            if (polar_vel_dist(pts[static_cast<std::size_t>(i)], q, vel_scale) <= 1.f)
                member[static_cast<std::size_t>(j)] = 1;
        }
    }

    float range_sum = 0.f, rate_sum = 0.f, sin_sum = 0.f, cos_sum = 0.f;
    float r_near = std::numeric_limits<float>::max();
    int representative = -1;
    std::vector<int> members;
    for (int i = 0; i < n; ++i) {
        if (!member[static_cast<std::size_t>(i)]) continue;
        const auto& p = pts[static_cast<std::size_t>(i)];
        range_sum += p.range_m;
        rate_sum += p.range_rate;
        sin_sum += std::sin(p.azimuth_rad);
        cos_sum += std::cos(p.azimuth_rad);
        if (p.range_m < r_near) { r_near = p.range_m; representative = i; }
        members.push_back(i);
    }
    const float inv_n = 1.f / static_cast<float>(members.size());
    out = {range_sum * inv_n, std::atan2(sin_sum, cos_sum), rate_sum * inv_n,
           representative, std::move(members)};
    return representative;
}

bool cluster_on_path(const RadarCluster& c, const PathPoly& path, float path_buf)
{
    if (!path.valid) return false;
    const float x = c.range_m * std::cos(c.azimuth_rad);
    const float y = c.range_m * std::sin(c.azimuth_rad);
    if (x < 0.5f || x > path.x_max_m) return false;
    return std::abs(y - path_y(path, x)) <= path_buf;
}

bool cv_proximal(float cluster_r, float cv_r)
{
    if (!(cv_r > 1.f) || !std::isfinite(cv_r)) return false;
    return std::abs(cluster_r - cv_r) <= std::max(8.f, 0.15f * cv_r);
}

int closest_of(const std::vector<RadarCluster>& clusters, const std::vector<int>& ids)
{
    int best = -1;
    float best_r = std::numeric_limits<float>::max();
    for (int i : ids) {
        const float r = clusters[static_cast<std::size_t>(i)].range_m;
        if (r < best_r) { best_r = r; best = i; }
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

    static constexpr float CIPO_PROB_MIN = 0.40f;  // below this → AD doesn't confirm CIPO

    // The probability gate exists to answer "is there a CIPO at all". An L1/L2
    // box already answers it, so when AutoSpeed sees one we take AutoDrive's
    // distance whatever its flag probability says — the flag and the distance
    // are separate heads, and a flag at 0.34 has been paired with a distance
    // that was right. With no box, AutoDrive's own confidence is all we have.
    const bool as_cipo_box =
        autospeed.valid &&
        std::any_of(autospeed.detections.begin(), autospeed.detections.end(),
                    [](const models::Detection& d) {
                        return d.class_id == 1 || d.class_id == 2;
                    });
    const bool ad_cipo_confirmed =
        autodrive.valid && (as_cipo_box || autodrive.flag_prob >= CIPO_PROB_MIN);

    // Range prior is AutoDrive only. Particle mean is not a CV measurement, so
    // it must not confirm a static rail at whatever the track already believes.
    const float* ad_ptr = nullptr;
    float ad_dist = 0.f;
    if (ad_cipo_confirmed) {
        ad_dist = cfg_.d_max_m * (1.f - autodrive.dist_normalized);
        if (ad_dist > 1.f) ad_ptr = &ad_dist;
    }

    Meas radar_meas;
    std::vector<int> match_members;
    if (cfg_.radar_enabled && radar) {
        const std::vector<models::Detection> no_dets;
        const auto& dets = autospeed.valid ? autospeed.detections : no_dets;
        const auto sel = select_cipo_radar(dets, *radar, path, ego_speed_ms, ad_ptr);
        match_members       = sel.members;
        radar_meas          = sel.meas;
        est.cut_in_detected = sel.cut_in;
        est.radar.fov_valid = sel.fov_valid;
        est.radar.fov_az_rad = sel.fov_az_rad;
        est.radar.match_i   = sel.match_i;
        est.radar.hit       = sel.hit;
        est.radar_scenario  = sel.scenario;
        if (radar_meas.valid) {
            est.radar_meas_valid = true;
            est.radar_dist_m     = radar_meas.distance_m;
            est.radar_vel_ms     = radar_meas.velocity_ms;
            est.radar.match_range_m = radar_meas.distance_m;
            est.radar.match_rate_ms = radar_meas.velocity_ms;
            est.radar.in_match.assign(radar->size(), 0);
            for (const int i : match_members)
                if (i >= 0 && i < static_cast<int>(radar->size()))
                    est.radar.in_match[static_cast<std::size_t>(i)] = 1;
            est.cipo_raw_found   = true;
            est.cipo_raw_dist_m  = radar_meas.distance_m;
        }
    } else if (as_h.valid) {
        est.cipo_raw_found  = true;
        est.cipo_raw_dist_m = as_h.distance_m;
    }

    const Meas& cipo_raw = cfg_.radar_enabled ? radar_meas : as_h;

    // ── Step 3: AutoDrive distance (gated by CIPO probability) ───────────────
    static constexpr float D_MAX = 150.f;

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

    // The homography inverts d = f·h / (v − v_horizon), so a fixed pixel error
    // on the box bottom maps to a distance error growing with d²: the two
    // pixels worth 0.3 m at 30 m are worth 8 m at 150 m. A flat noise figure
    // therefore states long-range readings far more confidently than it has
    // any right to, and a 155 m reading outvoted a 90 m one that was measured
    // far better. Grow the noise the way the geometry does.
    if (sel.meas.valid) {
        const float s = sel.meas.distance_m / cfg_.homography_ref_range_m;
        sel.meas.stddev_m = std::clamp(cfg_.cipo_noise_m * s * s,
                                       0.25f * cfg_.cipo_noise_m, 200.f);
    }
    return sel;
}

LongitudinalFusion::CIPOSelection
LongitudinalFusion::select_cipo_radar(const std::vector<models::Detection>& dets,
                                      const std::vector<RadarPoint>& radar,
                                      const PathPoly* path,
                                      const float* ego_speed_ms,
                                      const float* range_prior_m) const
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
        sel.members           = c.members;
        sel.scenario          = scenario;
        sel.hit               = hit;
        return sel;
    };

    float ego = 0.f;
    bool have_ego = false;
    if (ego_speed_ms && std::isfinite(*ego_speed_ms) && *ego_speed_ms > 0.5f) {
        ego = *ego_speed_ms;
        have_ego = true;
    } else {
        ego = estimate_ego_ms(radar);
        have_ego = ego > 1.f;
    }

    const auto clusters = cluster_radar(radar, cfg_.radar_max_range_m, 1.0f, ego, have_ego);

    std::vector<models::Detection> cipo_boxes;
    for (const auto& d : dets)
        if (d.class_id == 1 || d.class_id == 2) cipo_boxes.push_back(d);
    if (!cipo_boxes.empty())
        std::sort(cipo_boxes.begin(), cipo_boxes.end(),
                  [](const models::Detection& a, const models::Detection& b) {
                      return (a.y1 + a.y2) > (b.y1 + b.y2);
                  });

    const bool have_box = !cipo_boxes.empty();
    float az = 0.f, az_min = 0.f, az_max = 0.f, d_h = 0.f;
    bool cut_in = false;
    if (have_box) {
        const auto& box = cipo_boxes.front();
        cut_in = box.class_id == 2;
        az = bbox_u_to_radar_az((box.x1 + box.x2) * 0.5f, cfg_);
        const float az_a = bbox_u_to_radar_az(box.x1, cfg_);
        const float az_b = bbox_u_to_radar_az(box.x2, cfg_);
        const float az_pad = 0.2f * std::abs(az_a - az_b) + 0.005f;
        az_min = std::min(az_a, az_b) - az_pad;
        az_max = std::max(az_a, az_b) + az_pad;
        d_h = project_dist(H_, (box.x1 + box.x2) * 0.5f, box.y2);
    }

    constexpr float kBoxLatM = 1.75f;
    const auto near_cv = [&](const RadarCluster& c) {
        const bool ad_ok = range_prior_m && cv_proximal(c.range_m, *range_prior_m);
        const bool as_ok = cv_proximal(c.range_m, d_h);
        return ad_ok || as_ok;
    };

    // Cluster everything, keep in-path movers plus CV-confirmed statics,
    // then the closest of that list is the CIPO. The L1/L2 box is not a
    // search window; it only supplies the AutoSpeed range for statics.
    if (path && path->valid) {
        const float lat_zone = cfg_.radar_path_buffer_m;
        std::vector<int> candidates;
        for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
            const auto& c = clusters[static_cast<std::size_t>(i)];
            if (!cluster_on_path(c, *path, lat_zone)) continue;
            if (cluster_is_moving(c, ego, have_ego))
                candidates.push_back(i);
            else if (c.members.size() >= 2 && near_cv(c))
                candidates.push_back(i);
        }

        const int ic = closest_of(clusters, candidates);
        if (ic >= 0)
            return fill_c(clusters[static_cast<std::size_t>(ic)], cut_in,
                          RadarHit::Path, have_box ? 1 : 3, have_box, az);

        CIPOSelection miss;
        miss.fov_valid  = have_box;
        miss.fov_az_rad = az;
        return miss;
    }

    if (have_box) {
        const float prior = range_prior_m ? *range_prior_m : d_h;
        const bool  prior_ok = std::isfinite(prior) && prior > 1.f &&
                               prior < cfg_.radar_max_range_m;
        const float band = std::max(10.f, 0.25f * prior);
        RadarCluster mc;
        if (cluster_in_window(radar,
                              az_min, az_max,
                              prior_ok ? std::max(1.f, prior - band) : 0.f,
                              prior_ok ? prior + band : cfg_.radar_max_range_m,
                              az, kBoxLatM, 1.0f, mc) >= 0)
            return fill_c(mc, cut_in, RadarHit::Fov, 1, true, az);
        CIPOSelection miss;
        miss.fov_valid  = true;
        miss.fov_az_rad = az;
        return miss;
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
