#include "yaskawa_arm.hpp"
#include "yaskawa_arm_config.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <boost/asio/io_context.hpp>
#include <chrono>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/format.hpp>
#include <boost/io/ostream_joiner.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm.hpp>

#include <viam/lib/robot_socket.hpp>
#include <viam/lib/scope_guard.hpp>
#include <viam/sdk/common/mesh.hpp>
#include <viam/sdk/common/pose.hpp>
#include <viam/sdk/common/proto_value.hpp>
#include <viam/sdk/components/component.hpp>
#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/log/logging.hpp>
#include <viam/sdk/module/module.hpp>
#include <viam/sdk/module/service.hpp>
#include <viam/sdk/registry/registry.hpp>
#include <viam/sdk/rpc/grpc_context_observer.hpp>

#include <grpcpp/server_context.h>

#include <third_party/trajectories/Trajectory.h>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <ranges>
#include <viam/trajex/totg/tools/planner.hpp>
#include <viam/trajex/totg/totg.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_utils.hpp>
#include <viam/trajex/types/hertz.hpp>

#include "tcp_limit.hpp"
#include "utils.hpp"

// this chunk of code uses the rust FFI to handle the spatialmath calculations
// to turn a UR vector to a pose
extern "C" void* quaternion_from_euler_angles(double rx, double ry, double rz);
extern "C" void free_quaternion_memory(void* q);

extern "C" void* orientation_vector_from_quaternion(void* q);
extern "C" void free_orientation_vector_memory(void* ov);

extern "C" double* orientation_vector_get_components(void* ov);
extern "C" void free_orientation_vector_components(double* ds);

namespace {

constexpr double k_min_sampling_freq_hz = 1.0;
constexpr double k_max_sampling_freq_hz = 250.0;
constexpr double k_default_min_timestep_sec = 1e-2;

// --- URDF mesh loading -----------------------------------------------------------------------
//
// When get_kinematics serves a URDF, the Viam server resolves each <mesh> collision reference
// through its own normalizeURDFMeshPath ("package://<pkg>/rest" -> "rest") and looks the result
// up in the response's meshes_by_urdf_filepath map. An empty map makes the server fail geometry
// conversion with "mesh geometry requires mesh map to be provided". So we load every referenced
// mesh and key it exactly the way the server will look it up.

// Mirror of RDK referenceframe.normalizeURDFMeshPath: strip "package://" and the package segment.
std::string normalize_urdf_mesh_path(std::string path) {
    constexpr char k_prefix[] = "package://";
    constexpr std::size_t k_prefix_len = sizeof(k_prefix) - 1;
    if (path.compare(0, k_prefix_len, k_prefix) == 0) {
        path.erase(0, k_prefix_len);
        if (const auto slash = path.find('/'); slash != std::string::npos) {
            path.erase(0, slash + 1);
        }
    }
    return path;
}

// RDK accepts mesh content types "stl" and "ply"; derive it from the file extension (lowercased).
std::string mesh_content_type(const std::string& key) {
    const auto dot = key.find_last_of('.');
    std::string ext = (dot == std::string::npos) ? std::string{} : key.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

// Parse every <mesh filename="..."> reference out of the URDF text and load the corresponding file
// from kinematics_dir, keyed as the server resolves it. Throws if a referenced mesh file is absent.
std::map<std::string, viam::sdk::mesh> load_urdf_meshes(const std::string& urdf_text,
                                                        const std::filesystem::path& kinematics_dir) {
    std::map<std::string, viam::sdk::mesh> meshes;
    constexpr char k_attr[] = "filename=\"";
    constexpr std::size_t k_attr_len = sizeof(k_attr) - 1;

    for (auto pos = urdf_text.find(k_attr); pos != std::string::npos; pos = urdf_text.find(k_attr, pos)) {
        pos += k_attr_len;
        const auto end = urdf_text.find('"', pos);
        if (end == std::string::npos) {
            break;
        }
        const std::string raw = urdf_text.substr(pos, end - pos);
        pos = end + 1;

        const std::string key = normalize_urdf_mesh_path(raw);
        const std::string content_type = mesh_content_type(key);
        // Only ship actual mesh geometry; ignore any non-mesh filename attributes.
        if (content_type != "stl" && content_type != "ply") {
            continue;
        }
        if (meshes.count(key) != 0) {
            continue;
        }

        const auto mesh_path = kinematics_dir / key;
        std::ifstream mesh_file(mesh_path, std::ios::binary);
        if (!mesh_file) {
            throw std::runtime_error(
                boost::str(boost::format("URDF references mesh '%1%' but no file exists at '%2%'") % raw % mesh_path.string()));
        }
        std::vector<unsigned char> data(std::istreambuf_iterator<char>(mesh_file), {});
        if (mesh_file.bad()) {
            throw std::runtime_error(boost::str(boost::format("error reading mesh file '%1%'") % mesh_path.string()));
        }
        meshes.emplace(key, viam::sdk::mesh{content_type, std::move(data)});
    }
    return meshes;
}

xt::xarray<double> eigen_waypoints_to_xarray(const std::list<Eigen::VectorXd>& waypoints) {
    if (waypoints.empty()) {
        return xt::xarray<double>::from_shape({0, 0});
    }

    const size_t num_waypoints = waypoints.size();
    const size_t dof = static_cast<size_t>(waypoints.front().size());

    xt::xarray<double> result = xt::zeros<double>({num_waypoints, dof});

    size_t i = 0;
    for (const auto& waypoint : waypoints) {
        for (size_t j = 0; j < dof; ++j) {
            result(i, j) = waypoint[static_cast<Eigen::Index>(j)];
        }
        ++i;
    }

    return result;
}

// Build a trajectory_point_t from dynamic-sized position/velocity containers.
template <typename Container>
trajectory_point_t make_trajectory_point(const Container& positions, const Container& velocities, duration_t time_from_start) {
    trajectory_point_t pt{};
    const auto n = std::min(static_cast<size_t>(NUMBER_OF_DOF), static_cast<size_t>(positions.size()));
    for (size_t i = 0; i < n; ++i) {
        pt.positions[i] = positions(static_cast<Eigen::Index>(i));
    }
    const auto nv = std::min(static_cast<size_t>(NUMBER_OF_DOF), static_cast<size_t>(velocities.size()));
    for (size_t i = 0; i < nv; ++i) {
        pt.velocities[i] = velocities(static_cast<Eigen::Index>(i));
    }
    pt.time_from_start = time_from_start;
    return pt;
}

struct segment_accumulator {
    std::vector<trajectory_point_t> samples;
    std::chrono::duration<double> cumulative_time{0};
    double total_duration = 0.0;
    double total_generation_time = 0.0;
    size_t total_waypoints = 0;
    double total_arc_length = 0.0;
    size_t segment_count = 0;
};

template <typename Func>
void sampling_func(std::vector<trajectory_point_t>& samples, double duration_sec, double sampling_frequency_hz, const Func& f) {
    if (duration_sec <= 0.0 || sampling_frequency_hz <= 0.0) {
        throw std::invalid_argument("duration_sec and sampling_frequency_hz are not both positive");
    }
    static constexpr std::size_t k_max_samples = 2000000;
    const auto putative_samples = duration_sec * sampling_frequency_hz;
    if (!std::isfinite(putative_samples) || putative_samples > k_max_samples) {
        throw std::invalid_argument(
            "duration_sec and sampling_frequency_hz exceed "
            "the maximum allowable samples");
    }
    const auto num_samples = static_cast<std::size_t>(std::ceil(putative_samples) + 1);
    const double step = duration_sec / static_cast<double>((num_samples - 1));

    for (std::size_t i = 1; i < num_samples - 1; ++i) {
        samples.push_back(f(static_cast<double>(i) * step, step));
    }

    samples.push_back(f(duration_sec, step));
}

pose cartesian_position_to_pose(CartesianPosition&& pos) {
    auto q = std::unique_ptr<void, decltype(&free_quaternion_memory)>(
        quaternion_from_euler_angles(degrees_to_radians(pos.rx), degrees_to_radians(pos.ry), degrees_to_radians(pos.rz)),
        &free_quaternion_memory);
    auto ov = std::unique_ptr<void, decltype(&free_orientation_vector_memory)>(orientation_vector_from_quaternion(q.get()),
                                                                               &free_orientation_vector_memory);

    auto components = std::unique_ptr<double[], decltype(&free_orientation_vector_components)>(orientation_vector_get_components(ov.get()),
                                                                                               &free_orientation_vector_components);
    auto position = coordinates{pos.x, pos.y, pos.z};
    auto orientation = pose_orientation{components[0], components[1], components[2]};
    auto theta = radians_to_degrees(components[3]);

    return {position, orientation, theta};
}

std::vector<std::string> validate_config_(const ResourceConfig& cfg) {
    if (!find_config_attribute<std::string>(cfg, "host")) {
        throw std::invalid_argument("attribute `host` is required");
    }
    auto vel_dim = validate_joint_limit_attribute(cfg, "speed_rad_per_sec");
    auto acc_dim = validate_joint_limit_attribute(cfg, "acceleration_rad_per_sec2");
    if (vel_dim > 1 && acc_dim > 1 && vel_dim != acc_dim) {
        throw std::invalid_argument(
            std::format("`speed_rad_per_sec` ({} elements) and `acceleration_rad_per_sec2` ({} elements) "
                        "must have the same dimension",
                        vel_dim,
                        acc_dim));
    }

    auto threshold = find_config_attribute<double>(cfg, "reject_move_request_threshold_rad");
    constexpr double k_min_threshold = 0.0;
    constexpr double k_max_threshold = 2 * M_PI;
    if (threshold && (*threshold < k_min_threshold || *threshold > k_max_threshold)) {
        throw std::invalid_argument(
            std::format("attribute `reject_move_request_threshold_rad` should be "
                        "between {} and {} , it is : {} radians",
                        k_min_threshold,
                        k_max_threshold,
                        *threshold));
    }
    auto sampling_freq = find_config_attribute<double>(cfg, "trajectory_sampling_freq_hz");
    if (sampling_freq && (*sampling_freq < k_min_sampling_freq_hz || *sampling_freq > k_max_sampling_freq_hz)) {
        throw std::invalid_argument(
            boost::str(boost::format("attribute `trajectory_sampling_freq_hz` should be between %1% and %2%, it is: %3% Hz") %
                       k_min_sampling_freq_hz % k_max_sampling_freq_hz % *sampling_freq));
    }

    auto waypoint_dedup_tolerance = find_config_attribute<double>(cfg, "waypoint_deduplication_tolerance_deg");
    constexpr double k_min_waypoint_dedup_tolerance_deg = 0.0;
    constexpr double k_max_waypoint_dedup_tolerance_deg = 10.0;
    if (waypoint_dedup_tolerance && (*waypoint_dedup_tolerance < k_min_waypoint_dedup_tolerance_deg ||
                                     *waypoint_dedup_tolerance > k_max_waypoint_dedup_tolerance_deg)) {
        throw std::invalid_argument(
            std::format("attribute `waypoint_deduplication_tolerance_deg` should be between {} and {}, it is: {} degrees",
                        k_min_waypoint_dedup_tolerance_deg,
                        k_max_waypoint_dedup_tolerance_deg,
                        *waypoint_dedup_tolerance));
    }

    auto path_tolerance = find_config_attribute<double>(cfg, "path_tolerance_rad");
    if (path_tolerance && (*path_tolerance < 0.0 || *path_tolerance > 3.0)) {
        throw std::invalid_argument(std::format("attribute `path_tolerance_rad` must be between 0.0 and 3.0, got {}", *path_tolerance));
    }

    auto segmentation_threshold = find_config_attribute<double>(cfg, "segmentation_threshold_rad");
    if (segmentation_threshold && (*segmentation_threshold <= 0.0 || *segmentation_threshold > 0.1)) {
        throw std::invalid_argument(
            std::format("attribute `segmentation_threshold_rad` must be between (0.0, 0.1], got {}", *segmentation_threshold));
    }

    auto collinearization = find_config_attribute<double>(cfg, "collinearization_ratio");
    if (collinearization && (*collinearization < 0.0 || *collinearization > 2.0)) {
        throw std::invalid_argument(
            std::format("attribute `collinearization_ratio` must be between 0.0 and 2.0, got {}", *collinearization));
    }

    // Validate telemetry_output_path if provided
    auto telemetry_path = find_config_attribute<std::string>(cfg, "telemetry_output_path");
    if (telemetry_path && telemetry_path->empty()) {
        throw std::invalid_argument("attribute `telemetry_output_path` cannot be empty");
    }

    auto group_index = find_config_attribute<double>(cfg, "group_index");
    constexpr int k_min_group_index = 0;
    constexpr int k_max_group_index = MAX_GROUPS - 1;
    if (group_index && (*group_index < k_min_group_index || *group_index > k_max_group_index || floor(*group_index) != *group_index)) {
        throw std::invalid_argument(std::format("attribute `group_index` should be a whole number between {} and {} , it is : {}",
                                                k_min_group_index,
                                                k_max_group_index,
                                                *group_index));
    }

    return {};
}

using viam::make_scope_guard;

}  // namespace

const ModelFamily& YaskawaArm::model_family() {
    static const auto family = ModelFamily{"viam", "yaskawa-robots"};
    return family;
}

Model YaskawaArm::model(std::string model_name) {
    return {model_family(), std::move(model_name)};
}

std::vector<std::shared_ptr<ModelRegistration>> YaskawaArm::create_model_registrations(boost::asio::io_context& io_context) {
    const auto model_strings = {
        "gp12",
        "gp180-120",
        "gp35l",
    };

    const auto arm = API::get<Arm>();
    const auto registration_factory = [&](auto m) {
        const auto model = YaskawaArm::model(m);
        return std::make_shared<ModelRegistration>(
            arm,
            model,
            // NOLINTNEXTLINE(performance-unnecessary-value-param): Signature is
            // fixed by ModelRegistration.
            [&, model](const auto& deps, const auto& config) {
                auto robot = std::make_shared<YaskawaArm>(model, deps, config, io_context);
                robot->configure(deps, config);
                return robot;
            },
            [](auto const& config) { return validate_config_(config); });
    };

    auto registrations = model_strings | boost::adaptors::transformed(registration_factory);
    return {std::make_move_iterator(begin(registrations)), std::make_move_iterator(end(registrations))};
}

YaskawaArm::YaskawaArm(Model model, const Dependencies&, const ResourceConfig& cfg, boost::asio::io_context& io_context)
    : Arm(cfg.name()), model_(std::move(model)), io_context_(io_context) {
    // Configure the global logger to use VIAM SDK logging
}

void YaskawaArm::configure(const Dependencies& deps, const ResourceConfig& cfg) {
    configure_logger(cfg);

    VIAM_SDK_LOG(info) << "Yaskawa Arm constructor called (model: " << model_.to_string() << ")";

    configure_(deps, cfg);
}

void YaskawaArm::configure_(const Dependencies&, const ResourceConfig& config) {
    VIAM_SDK_LOG(info) << "Yaskawa arm  starting up";

    const auto module_executable_path = boost::dll::program_location();
    const auto module_executable_directory = module_executable_path.parent_path();

    resource_root_ = std::filesystem::canonical(module_executable_directory / k_relpath_bindir_to_datadir / "yaskawa-robots");
    VIAM_SDK_LOG(info) << "Yaskawa robots module executable found in `" << module_executable_path << "; resources will be found in `"
                       << resource_root_ << "`";

    // Only tear down the existing controller if the connection target actually changed. A
    // disconnect+reconstruct on every reconfigure wastes a TCP/UDP cycle (and breaks any
    // sibling arm sharing the same controller via the registry) when the user is only
    // tweaking unrelated attributes like speed limits.
    auto new_host_attr = find_config_attribute<std::string>(config, "host");
    if (!new_host_attr) {
        throw std::runtime_error("host attribute is required");
    }
    const auto& new_host = *new_host_attr;
    auto new_tcp_port_attr = find_config_attribute<double>(config, "tcp_port");
    auto new_tcp_port = new_tcp_port_attr ? static_cast<uint16_t>(*new_tcp_port_attr) : static_cast<uint16_t>(TCP_PORT);
    if (robot_ && (robot_->host() != new_host || robot_->tcp_port() != new_tcp_port)) {
        VIAM_SDK_LOG(info) << "connection target changed (" << robot_->host() << ":" << robot_->tcp_port() << " -> " << new_host << ":"
                           << new_tcp_port << "), resetting connection";
        robot_->disconnect();
        robot_.reset();
    }
    threshold_ = find_config_attribute<double>(config, "reject_move_request_threshold_rad");

    // Get telemetry output path from config or fall back to VIAM_MODULE_DATA
    telemetry_output_path_ = [&] {
        auto path = find_config_attribute<std::string>(config, "telemetry_output_path");
        if (path) {
            return path.value();
        }

        auto* const viam_module_data = std::getenv("VIAM_MODULE_DATA");  // NOLINT: Yes, we know getenv isn't thread safe
        if (!viam_module_data) {
            throw std::runtime_error("required environment variable `VIAM_MODULE_DATA` unset");
        }
        VIAM_SDK_LOG(debug) << "VIAM_MODULE_DATA: " << viam_module_data;

        return std::string{viam_module_data};
    }();

    VIAM_SDK_LOG(debug) << "telemetry output path set : " << telemetry_output_path_;

    if (!robot_) {
        robot_ = YaskawaController::get_or_create(io_context_, config);
    }
    group_index_ = static_cast<uint32_t>(find_config_attribute<double>(config, "group_index").value_or(0));

    auto dof = number_of_dof_configured(config, "speed_rad_per_sec", "acceleration_rad_per_sec2");
    velocity_limits_ = read_limit_vector(config, "speed_rad_per_sec", dof);
    acceleration_limits_ = read_limit_vector(config, "acceleration_rad_per_sec2", dof);

    // Build the kinematics model table for TCP-speed-limited moves, if this model ships a URDF.
    // Models without a URDF (e.g. gp180-120) stay joint-space-only; a tcp_speed request for
    // them throws later, in generate_trajectory_.
    model_table_tensor_.reset();
    const auto urdf_path = resource_root_ / "kinematics" / (model_.model_name() + ".urdf");
    if (std::filesystem::exists(urdf_path)) {
        try {
            model_table_tensor_ = model_table_tensor_from_urdf(urdf_path);
            VIAM_SDK_LOG(info) << "loaded model table from " << urdf_path << " ("
                               << model_table_tensor_->shape()[0] << " joints)";
        } catch (const std::exception& e) {
            VIAM_SDK_LOG(warn) << "failed to build model table from " << urdf_path << ": " << e.what()
                               << "; TCP speed limits will be unavailable for this arm";
        }
    } else {
        VIAM_SDK_LOG(info) << "no URDF at " << urdf_path << "; TCP speed limits unavailable for model "
                           << model_.model_name();
    }

    constexpr double k_default_trajectory_sampling_freq = 3.0;
    constexpr double k_default_waypoint_dedup_tolerance_rads = 1e-3;
    constexpr double k_default_segmentation_threshold = 0.005;

    trajectory_sampling_freq_ =
        find_config_attribute<double>(config, "trajectory_sampling_freq_hz").value_or(k_default_trajectory_sampling_freq);
    auto waypoint_dedup_tolerance_deg = find_config_attribute<double>(config, "waypoint_deduplication_tolerance_deg");
    waypoint_dedup_tolerance_rad_ =
        waypoint_dedup_tolerance_deg ? degrees_to_radians(*waypoint_dedup_tolerance_deg) : k_default_waypoint_dedup_tolerance_rads;
    use_new_trajectory_planner_ = find_config_attribute<bool>(config, "enable_new_trajectory_planner").value_or(true);
    use_urdfs_ = find_config_attribute<bool>(config, "use_urdfs").value_or(false);

    // Default TCP speed cap applied to every move; a per-call MoveOptions.max_tcp_speed overrides it.
    config_max_tcp_speed_ = find_config_attribute<double>(config, "max_tcp_speed_m_per_sec");
    if (config_max_tcp_speed_ && (!std::isfinite(*config_max_tcp_speed_) || *config_max_tcp_speed_ <= 0.0)) {
        throw std::invalid_argument("max_tcp_speed_m_per_sec must be a positive, finite number of meters per second");
    }
    path_tolerance_rad_ = find_config_attribute<double>(config, "path_tolerance_rad").value_or(0.1);
    collinearization_ratio_ = find_config_attribute<double>(config, "collinearization_ratio");
    segmentation_threshold_rad_ =
        find_config_attribute<double>(config, "segmentation_threshold_rad").value_or(k_default_segmentation_threshold);

    // YaskawaController spins up its FSM in the constructor; the resource is up immediately,
    // and the FSM connects (and reconnects) in the background.
}

std::vector<double> YaskawaArm::get_joint_positions(const ProtoStruct&) {
    const std::shared_lock rlock{config_mutex_};
    auto joint_rads = robot_->get_group_position_velocity_torque(static_cast<uint8_t>(group_index_));
    auto joint_position_degree = joint_rads.position | boost::adaptors::transformed(radians_to_degrees<const double&>);
    return {std::begin(joint_position_degree), std::end(joint_position_degree)};
}

void YaskawaArm::move_through_joint_positions(const std::vector<std::vector<double>>& positions,
                                              const MoveOptions& options,
                                              const viam::sdk::ProtoStruct&) {
    const std::shared_lock rlock{config_mutex_};

    if (positions.empty()) {
        throw std::invalid_argument("positions must not be empty");
    }

    std::list<Eigen::VectorXd> waypoints;

    // Convert joint positions from degrees to radians and filter duplicate
    // waypoints
    for (const auto& position : positions) {
        auto next_waypoint_deg = Eigen::VectorXd::Map(position.data(), boost::numeric_cast<Eigen::Index>(position.size()));
        auto next_waypoint_rad = degrees_to_radians(std::move(next_waypoint_deg)).eval();

        // Skip waypoints that are too close to the previous one to avoid
        // redundant motion
        if ((!waypoints.empty()) && (next_waypoint_rad.isApprox(waypoints.back(), waypoint_dedup_tolerance_rad_))) {
            continue;
        }
        waypoints.emplace_back(std::move(next_waypoint_rad));
    }

    auto velocity = velocity_limits_;
    if (options.max_vel_degs_per_sec) {
        apply_move_limit(velocity, *options.max_vel_degs_per_sec);
    }

    auto acceleration = acceleration_limits_;
    if (options.max_acc_degs_per_sec2) {
        apply_move_limit(acceleration, *options.max_acc_degs_per_sec2);
    }

    auto tcp_speed = read_tcp_speed(options);
    if (!tcp_speed) {
        // No per-call override; fall back to the configured default cap, if any.
        tcp_speed = config_max_tcp_speed_;
    }
    if (tcp_speed && !use_new_trajectory_planner_) {
        // The TCP speed cap is enforced only by the totg planner. Refuse rather than run an
        // uncapped move on the legacy generator.
        throw std::invalid_argument(
            "max_tcp_speed requires the new trajectory planner; set enable_new_trajectory_planner=true");
    }

    const auto unix_time = unix_time_iso8601();

    std::optional<RealtimeTrajectoryLogger> logger;
    try {
        logger.emplace(telemetry_output_path_, unix_time, model_.model_name(), group_index_);
    } catch (const std::exception& e) {
        VIAM_SDK_LOG(warn) << "Failed to create realtime trajectory logger: " << e.what();
    }

    auto traj_result = generate_trajectory_(waypoints, group_index_, unix_time, velocity, acceleration, tcp_speed, logger);
    if (!traj_result) {
        VIAM_SDK_LOG(debug) << "already at desired position";
        return;
    }

    robot_
        ->enqueue_move_request(
            group_index_,
            static_cast<uint32_t>(velocity.size()),
            std::move(traj_result->samples),
            traj_result->tolerance,
            trajectory_sampling_freq_,
            std::move(logger),
            [observer = viam::sdk::GrpcContextObserver::current()] { return observer && observer->context().IsCancelled(); })
        .get();
}

void YaskawaArm::move_to_joint_positions(const std::vector<double>& positions, const ProtoStruct&) {
    const std::shared_lock rlock{config_mutex_};

    auto next_waypoint_deg = Eigen::VectorXd::Map(positions.data(), boost::numeric_cast<Eigen::Index>(positions.size())).eval();
    auto next_waypoint_rad = degrees_to_radians(std::move(next_waypoint_deg));
    std::list<Eigen::VectorXd> waypoints;
    waypoints.emplace_back(std::move(next_waypoint_rad));

    const auto unix_time = unix_time_iso8601();

    std::optional<RealtimeTrajectoryLogger> logger;
    try {
        logger.emplace(telemetry_output_path_, unix_time, model_.model_name(), group_index_);
    } catch (const std::exception& e) {
        VIAM_SDK_LOG(warn) << "Failed to create realtime trajectory logger: " << e.what();
    }

    auto traj_result =
        generate_trajectory_(waypoints, group_index_, unix_time, velocity_limits_, acceleration_limits_, std::nullopt, logger);
    if (!traj_result) {
        VIAM_SDK_LOG(debug) << "already at desired position";
        return;
    }

    robot_
        ->enqueue_move_request(
            group_index_,
            static_cast<uint32_t>(velocity_limits_.size()),
            std::move(traj_result->samples),
            traj_result->tolerance,
            trajectory_sampling_freq_,
            std::move(logger),
            [observer = viam::sdk::GrpcContextObserver::current()] { return observer && observer->context().IsCancelled(); })
        .get();
}

::viam::sdk::KinematicsData YaskawaArm::get_kinematics(const ProtoStruct&) {
    using KinematicsDataSVA = viam::sdk::KinematicsDataSVA;
    using KinematicsDataURDF = viam::sdk::KinematicsDataURDF;
    const std::shared_lock rlock{config_mutex_};

    // use_urdfs selects which on-disk representation we serve: the URDF or the SVA JSON.
    const char* const extension = use_urdfs_ ? "urdf" : "json";
    const auto kinematics_file_path =
        resource_root_ / str(boost::format("kinematics/%1%.%2%") % model_.model_name() % extension);

    // Open the file in binary mode
    std::ifstream kinematics_file(kinematics_file_path, std::ios::binary);
    if (!kinematics_file) {
        if (use_urdfs_) {
            throw std::runtime_error(boost::str(
                boost::format("use_urdfs=true but no URDF kinematics file exists for model '%1%' (looked for '%2%')") %
                model_.model_name() % kinematics_file_path.string()));
        }
        throw std::runtime_error(boost::str(boost::format("unable to open kinematics file '%1%'") % kinematics_file_path.string()));
    }

    // Read the entire file into a vector without computing size ahead of time
    std::vector<char> temp_bytes(std::istreambuf_iterator<char>(kinematics_file), {});
    if (kinematics_file.bad()) {
        throw std::runtime_error(boost::str(boost::format("error reading kinematics file '%1%'") % kinematics_file_path.string()));
    }

    // Convert to unsigned char vector
    if (use_urdfs_) {
        KinematicsDataURDF urdf({temp_bytes.begin(), temp_bytes.end()});
        // The URDF declares collision geometry via package:// mesh references; the server needs the
        // actual STL bytes alongside the URDF, or it fails geometry conversion ("mesh map ... nil").
        const std::string urdf_text(temp_bytes.begin(), temp_bytes.end());
        urdf.meshes_by_urdf_filepath = load_urdf_meshes(urdf_text, resource_root_ / "kinematics");
        return urdf;
    }
    return KinematicsDataSVA({temp_bytes.begin(), temp_bytes.end()});
}

pose YaskawaArm::get_end_position(const ProtoStruct&) {
    const std::shared_lock rlock{config_mutex_};

    auto p = cartesian_position_to_pose(robot_->getCartPosition(group_index_));

    // The controller is what provides is with a cartesian position. However, it
    // is not aware of the base links position, i.e. that is translated up by some
    // amount.
    const auto model_name = model_.model_name();
    if (model_name == "gp12") {
        // For the gp12 model that translation is 450mm
        // https://github.com/ros-industrial/motoman/blob/noetic-devel/motoman_gp12_support/urdf/gp12_macro.xacro#L154
        p.coordinates.z += 450;
    } else if (model_name == "gp180-120") {
        // For the gp180-120 model that translation is 650mm
        // https://github.com/Yaskawa-Global/motoman_ros2_support_packages/blob/3187e27b9e59615b7bb4d25ca406e4280e8ebe26/motoman_gp180_support/urdf/gp180_120_macro.xacro#L148
        p.coordinates.z += 650;
    } else if (model_name == "gp35l") {
        // For the gp35l model that translation is 540mm (from gp35l_macro.xacro
        // joint_1_s origin xyz="0 0 0.540")
        p.coordinates.z += 540;
    } else {
        VIAM_SDK_LOG(warn) << "No pose offset applied for model: " << model_name;
    }

    return p;
}

void YaskawaArm::stop(const ProtoStruct&) {
    if (!robot_->stop(group_index_)) {
        // we were not checking this before. Add a log to see how often it occurs
        VIAM_SDK_LOG(warn) << "stop did not error but did not return as stopped";
    }
}

ProtoStruct YaskawaArm::do_command(const ProtoStruct&) {
    ProtoStruct resp = ProtoStruct{};
    return resp;
}

ProtoStruct YaskawaArm::get_status() {
    return ProtoStruct{};
}

std::optional<YaskawaArm::TrajectoryResult> YaskawaArm::generate_trajectory_(const std::list<Eigen::VectorXd>& waypoints,
                                                                             uint32_t group_index,
                                                                             const std::string& unix_time,
                                                                             const Eigen::VectorXd& max_velocity_vec,
                                                                             const Eigen::VectorXd& max_acceleration_vec,
                                                                             std::optional<double> tcp_speed,
                                                                             std::optional<RealtimeTrajectoryLogger>& logger) {
    VIAM_SDK_LOG(info) << "move: start unix_time_ms " << unix_time << " waypoints size " << waypoints.size();

    const auto& original_waypoints = waypoints;

    auto log_failure = [&](const std::string& replay_json) {
        auto filename = std::format("{}/{}_failed_trajectory.trajex-totg-replay.json", telemetry_output_path_, unix_time);
        std::ofstream json_file(filename);
        json_file << replay_json;
    };

    using namespace viam::trajex;

    totg::planner_base::config planner_cfg;
    planner_cfg.velocity_limits = xt::xarray<double>::from_shape({static_cast<size_t>(max_velocity_vec.size())});
    planner_cfg.acceleration_limits = xt::xarray<double>::from_shape({static_cast<size_t>(max_acceleration_vec.size())});
    std::ranges::copy(max_velocity_vec, planner_cfg.velocity_limits.begin());
    std::ranges::copy(max_acceleration_vec, planner_cfg.acceleration_limits.begin());
    planner_cfg.path_blend_tolerance = path_tolerance_rad_;
    planner_cfg.colinearization_ratio = collinearization_ratio_;
    planner_cfg.segment_totg = false;
    planner_cfg.tcp = make_tcp_limit(tcp_speed, model_table_tensor_, model_.model_name());
    // Record the model-table provenance so a failed-move replay record can reconstruct the TCP
    // jacobian. serialize_for_replay only writes it when a TCP limit is also set.
    planner_cfg.model_table = model_table_tensor_;

    auto planner =
        totg::planner<segment_accumulator>(planner_cfg)
            .with_waypoint_provider([&](auto& p) -> totg::waypoint_accumulator {
                auto curr_joint_pos = robot_->get_group_position_velocity_torque(static_cast<uint8_t>(group_index)).position;
                auto current_pos = p.stash(eigen_waypoints_to_xarray(
                    {Eigen::VectorXd::Map(curr_joint_pos.data(), boost::numeric_cast<Eigen::Index>(curr_joint_pos.size()))}));
                auto goal_waypoints = p.stash(eigen_waypoints_to_xarray(waypoints));

                totg::waypoint_accumulator acc(*current_pos);
                acc.add_waypoints(*goal_waypoints);
                return acc;
            })
            .with_waypoint_preprocessor([&](auto&, auto& acc) { acc = totg::deduplicate_waypoints(acc, waypoint_dedup_tolerance_rad_); })
            .with_segmenter(  // NOLINTNEXTLINE(performance-unnecessary-value-param)
                [&](auto&, totg::waypoint_accumulator acc) {
                    return totg::segment_at_reversals(std::move(acc), segmentation_threshold_rad_);
                });

    if (use_new_trajectory_planner_) {
        planner.with_totg(
            [&](const auto&,
                segment_accumulator& acc,
                const totg::waypoint_accumulator& seg,
                totg::trajectory&& traj,
                std::chrono::microseconds elapsed) {
                acc.total_waypoints += seg.size();
                acc.total_duration += traj.duration().count();
                acc.total_arc_length += static_cast<double>(traj.path().length());
                acc.total_generation_time += std::chrono::duration<double>(elapsed).count();
                ++acc.segment_count;

                if (acc.total_duration > 600) {
                    throw std::runtime_error("total trajectory duration exceeds maximum allowed duration");
                }

                auto sampler = totg::uniform_sampler::quantized_for_trajectory(traj, types::hertz{trajectory_sampling_freq_});

                const bool is_first_segment = (acc.segment_count == 1);
                for (const auto& sample : traj.samples(sampler) | std::views::drop(is_first_segment ? 0 : 1)) {
                    const auto absolute_time = acc.cumulative_time + std::chrono::duration<double>(sample.time.count());
                    auto secs = std::chrono::floor<std::chrono::seconds>(absolute_time);
                    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(absolute_time - secs);
                    acc.samples.push_back(
                        make_trajectory_point(sample.configuration,
                                              sample.velocity,
                                              {static_cast<uint32_t>(secs.count()), static_cast<uint32_t>(nanos.count())}));
                }

                acc.cumulative_time += std::chrono::duration<double>(traj.duration());

                VIAM_SDK_LOG(info) << "trajex/totg segment generated, waypoints: " << seg.size()
                                   << ", duration: " << traj.duration().count() << "s, samples: " << acc.samples.size()
                                   << ", arc length: " << traj.path().length();
            },
            [&](const auto& p, const segment_accumulator&, const totg::waypoint_accumulator& seg, const std::exception& e) {
                log_failure(p.serialize_for_replay(seg, e.what()));
            });
    }

    planner.with_legacy(
        [&](const auto&,
            segment_accumulator& acc,
            const totg::waypoint_accumulator& seg,
            Path&&,
            Trajectory&& traj,
            std::chrono::microseconds elapsed) {
            const double duration = traj.getDuration();

            if (!std::isfinite(duration)) {
                throw std::runtime_error("trajectory.getDuration() was not a finite number");
            }
            if (duration > 600) {
                throw std::runtime_error("trajectory.getDuration() exceeds 10 minutes");
            }
            if (duration < k_default_min_timestep_sec) {
                VIAM_SDK_LOG(debug) << "duration of move is too small, assuming arm is at goal";
                return;
            }

            if (acc.samples.empty()) {
                acc.samples.push_back(make_trajectory_point(traj.getPosition(0.0), traj.getVelocity(0.0), {0, 0}));
            }

            sampling_func(acc.samples, duration, trajectory_sampling_freq_, [&](const double t, const double) {
                const auto absolute_time = acc.cumulative_time + std::chrono::duration<double>(t);
                auto secs = std::chrono::floor<std::chrono::seconds>(absolute_time);
                auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(absolute_time - secs);
                return make_trajectory_point(
                    traj.getPosition(t), traj.getVelocity(t), {static_cast<uint32_t>(secs.count()), static_cast<uint32_t>(nanos.count())});
            });

            acc.cumulative_time += std::chrono::duration<double>(duration);
            acc.total_waypoints += seg.size();
            acc.total_duration += duration;
            acc.total_generation_time += std::chrono::duration<double>(elapsed).count();
            ++acc.segment_count;
        },
        [&](const auto& p, const segment_accumulator&, const totg::waypoint_accumulator& seg, const std::exception& e) {
            log_failure(p.serialize_for_replay(seg, e.what()));
        });

    auto result = planner.execute([&](const auto& p, auto trajex_out, auto legacy_out) -> std::optional<segment_accumulator> {
        if (trajex_out.receiver) {
            auto& acc = *trajex_out.receiver;
            VIAM_SDK_LOG(info) << "trajex/totg trajectory generated, total waypoints: " << acc.total_waypoints
                               << ", total duration: " << acc.total_duration << "s, total samples: " << acc.samples.size()
                               << ", total arc length: " << acc.total_arc_length << ", generation_time: " << acc.total_generation_time
                               << "s";
            return std::move(trajex_out.receiver);
        }

        // The TCP speed cap is enforced only by the totg planner. If a cap was requested but
        // totg produced no trajectory, refuse the legacy fallback (which ignores the cap) and
        // surface the failure instead of silently running an uncapped move.
        if (tcp_speed && legacy_out.receiver) {
            if (trajex_out.error) {
                std::rethrow_exception(trajex_out.error);
            }
            throw std::runtime_error(
                "max_tcp_speed was requested but the totg planner produced no trajectory; refusing "
                "the legacy fallback, which does not enforce the TCP speed limit");
        }

        if (legacy_out.receiver) {
            VIAM_SDK_LOG(info) << "trajectory generation uses legacy generator";
            return std::move(legacy_out.receiver);
        }

        if (legacy_out.error) {
            std::rethrow_exception(legacy_out.error);
        }
        if (trajex_out.error) {
            std::rethrow_exception(trajex_out.error);
        }

        if (p.processed_waypoint_count() < 2) {
            return std::nullopt;
        }
        throw std::runtime_error("both trajectory generators failed");
    });

    if (!result || result->samples.empty()) {
        return std::nullopt;
    }

    auto& samples = result->samples;

    if (logger.has_value()) {
        logger->set_max_velocity(max_velocity_vec);
        logger->set_max_acceleration(max_acceleration_vec);
        logger->set_waypoints(original_waypoints);
        logger->set_planned_trajectory(samples, static_cast<int>(max_velocity_vec.size()));

        // Record the planner inputs so a successful move can be re-planned and visualized,
        // including the TCP limit when one was applied.
        std::optional<std::vector<std::vector<double>>> model_table_rows;
        if (model_table_tensor_) {
            const auto& table = *model_table_tensor_;
            std::vector<std::vector<double>> rows(table.shape()[0], std::vector<double>(table.shape()[1]));
            for (size_t i = 0; i < table.shape()[0]; ++i) {
                for (size_t j = 0; j < table.shape()[1]; ++j) {
                    rows[i][j] = table(i, j);
                }
            }
            model_table_rows = std::move(rows);
        }
        logger->set_planner_options(path_tolerance_rad_, tcp_speed, model_table_rows);
    }

    VIAM_SDK_LOG(debug) << "total trajectory points: " << samples.size();

    return TrajectoryResult{std::move(samples), {}};
}

YaskawaArm::~YaskawaArm() {
    try {
        // robot_ is only created partway through configure_() (after resource_root_ resolution,
        // host validation, etc.). If construction threw before that point, the arm is destroyed
        // with robot_ still null — guard so the dtor reports the real error instead of segfaulting
        // on robot_->disconnect().
        if (robot_) {
            robot_->disconnect();
        }
    } catch (...) {
        const auto unconditional_abort = make_scope_guard([] { std::abort(); });
        try {
            throw;
        } catch (const std::exception& ex) {
            VIAM_SDK_LOG(error) << "YaskawaArm dtor failed with a std::exception - module service will terminate: " << ex.what();
        } catch (...) {
            VIAM_SDK_LOG(error) << "YaskawaArm dtor failed with an unknown exception - module service will terminate";
        }
    }
}

bool YaskawaArm::is_moving() {
    return robot_->get_robot_status().is_group_moving(group_index_);
}
