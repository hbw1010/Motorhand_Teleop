#include "ve33_model.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace motorhand::native {
namespace {
using V3 = Eigen::Vector3d;
using Transform = Eigen::Isometry3d;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tree = Eigen::Matrix<double, 68, 1>;
using Inputs = Eigen::Matrix<double, 12, 1>;
using Derivative = Eigen::Matrix<double, 68, 12>;
constexpr double inf = std::numeric_limits<double>::infinity();

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("ve33: " + message); }

V3 triple(const char* text, const V3& fallback) {
    if (!text) return fallback;
    std::istringstream in(text);
    V3 v;
    std::string extra;
    if (!(in >> v.x() >> v.y() >> v.z()) || (in >> extra) || !v.allFinite()) fail("invalid URDF vector");
    return v;
}
V3 triple(const YAML::Node& n) {
    if (!n.IsSequence() || n.size() != 3) fail("invalid closure vector");
    V3 v(n[0].as<double>(), n[1].as<double>(), n[2].as<double>());
    if (!v.allFinite()) fail("non-finite closure vector");
    return v;
}
V3 triple(const nlohmann::json& n) {
    if (!n.is_array() || n.size() != 3) fail("invalid landmark vector");
    V3 v(n[0].get<double>(), n[1].get<double>(), n[2].get<double>());
    if (!v.allFinite()) fail("non-finite landmark vector");
    return v;
}
nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) fail("cannot open " + path.string());
    return nlohmann::json::parse(f);
}
std::string attr(const tinyxml2::XMLElement* e, const char* name) {
    if (!e || !e->Attribute(name)) fail(std::string("missing URDF attribute ") + name);
    return e->Attribute(name);
}

template <typename T> void require_finite(const T& a) {
    for (double v : a) if (!std::isfinite(v)) fail("non-finite input");
}
}  // namespace

struct Ve33Model::Impl {
    struct Joint {
        std::string name, parent, child;
        int kind = 0, parent_node = -1, tree = -1;
        Transform origin = Transform::Identity();
        V3 axis = V3::UnitZ();
        double lower = -inf, upper = inf;
    };
    struct Witness {
        int a, b;
        V3 pa, pb;
    };
    struct Site { int link; V3 local; };
    struct Group {
        std::string name;
        std::vector<int> indices, inputs, witnesses;
        int passive = 0;
        Matrix passive_map, input_map;
        Vector lower, upper;
        std::vector<int> frame_nodes;
        std::vector<std::vector<unsigned char>> mask_a, mask_b;
    };
    struct Work {
        Tree state;
        bool has_state = false, has_derivative = false, has_validation = false;
        double max_gap = 0;
        Derivative derivative;
        std::vector<Transform> frames, scratch_frames;
    };
    std::vector<Joint> nodes;
    std::vector<int> topology;
    std::unordered_map<std::string, int> joint_node, child_node, tree_index;
    std::set<std::string> roots;
    std::array<int, 68> tree_node{};
    std::array<int, 12> motor_index{};
    std::vector<Witness> witnesses;
    std::vector<Group> groups;
    std::array<Site, 21> sites;
    Eigen::Matrix3d canonical_to_model;
    Eigen::Matrix<double, 12, 68> angles = Eigen::Matrix<double, 12, 68>::Zero();
    std::array<bool, 68> branch_mask{};
    double tolerance = 0;
    GlovePose zero_pose;

    int link(const std::string& name) const {
        auto it = child_node.find(name);
        if (it != child_node.end()) return it->second;
        if (roots.count(name)) return -1;
        fail("unknown link " + name);
    }
    bool ancestor(int node, int joint) const {
        while (node >= 0) {
            if (node == joint) return true;
            node = nodes[node].parent_node;
        }
        return false;
    }
    void fill_frames(const Tree& state, const std::vector<int>& order,
                     std::vector<Transform>& out) const {
        if (out.size() != nodes.size()) out.resize(nodes.size());
        for (int i : order) {
            const Joint& j = nodes[i];
            Transform local = j.origin;
            const double q = j.tree < 0 ? 0.0 : state[j.tree];
            if (j.kind == 1) local.rotate(Eigen::AngleAxisd(q, j.axis));
            else if (j.kind == 2) local.translate(q * j.axis);
            out[i] = j.parent_node < 0 ? local : out[j.parent_node] * local;
        }
    }
    std::vector<Transform> frames(const Tree& state) const {
        std::vector<Transform> out;
        fill_frames(state, topology, out);
        return out;
    }
    const std::vector<Transform>& cached_frames(const Tree& state, Work& work) const {
        if (!work.has_state || !(work.state.array() == state.array()).all()) {
            fill_frames(state, topology, work.frames);
            work.state = state;
            work.has_state = true;
            work.has_derivative = false;
            work.has_validation = false;
        }
        return work.frames;
    }
    static V3 world_point(int node, const V3& p, const std::vector<Transform>& f) {
        return node < 0 ? p : f[node] * p;
    }
    V3 world_axis(int node, const std::vector<Transform>& f) const {
        return f[node].linear() * nodes[node].axis;
    }
    V3 point_derivative(int node, const V3& p, const std::vector<Transform>& f) const {
        const V3 a = world_axis(node, f);
        return nodes[node].kind == 2 ? a : a.cross(p - f[node].translation()).eval();
    }
    void residual_jac(const std::vector<Transform>& f, const Group& group,
                      Vector& residual, Matrix& jac) const {
        residual.resize(3 * group.witnesses.size());
        jac.setZero(residual.size(), group.indices.size());
        for (int r = 0; r < static_cast<int>(group.witnesses.size()); ++r) {
            const Witness& w = witnesses[group.witnesses[r]];
            V3 a = world_point(w.a, w.pa, f), b = world_point(w.b, w.pb, f);
            residual.segment<3>(3 * r) = a - b;
            for (int c = 0; c < static_cast<int>(group.indices.size()); ++c) {
                int n = tree_node[group.indices[c]];
                if (group.mask_a[r][c]) jac.block<3, 1>(3*r, c) += point_derivative(n, a, f);
                if (group.mask_b[r][c]) jac.block<3, 1>(3*r, c) -= point_derivative(n, b, f);
            }
        }
    }
    double gap(const std::vector<Transform>& f) const {
        double result = 0;
        for (const auto& w : witnesses)
            result = std::max(result, (world_point(w.a, w.pa, f) - world_point(w.b, w.pb, f)).norm());
        return result;
    }
    void validate(const Tree& state, Work& work) const {
        if (!state.allFinite()) fail("non-finite tree state");
        const auto& f = cached_frames(state, work);
        if (work.has_validation) return;
        for (int i = 0; i < 68; ++i) {
            const auto& n = nodes[tree_node[i]];
            if (state[i] < n.lower || state[i] > n.upper) fail("tree limit exceeded: " + n.name);
        }
        double g = gap(f);
        if (!std::isfinite(g) || g > tolerance) fail("closure gap exceeds tolerance");
        work.max_gap = g;
        work.has_validation = true;
    }
    Matrix least_squares(const Matrix& a, const Matrix& b, const std::string& label) const {
        Eigen::JacobiSVD<Matrix> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& s = svd.singularValues();
        if (s.size() != a.cols() || s.size() == 0 || !s.allFinite() ||
            s[s.size()-1] <= s[0] * 1e-9 || s[0] <= 0) fail(label + " singular");
        svd.setThreshold(1e-9);
        Matrix x = svd.solve(b);
        if (!x.allFinite()) fail(label + " non-finite solution");
        return x;
    }
    const Derivative& derivatives(const Tree& state, Work& work) const {
        const auto& f = cached_frames(state, work);
        if (work.has_derivative) return work.derivative;
        Derivative d = Derivative::Zero();
        for (const auto& g : groups) {
            Vector r; Matrix j;
            residual_jac(f, g, r, j);
            Matrix a = j * g.passive_map, b = j * g.input_map;
            Matrix response = least_squares(a, -b, g.name + " derivative");
            if ((a * response + b).cwiseAbs().maxCoeff() > 1e-7) fail(g.name + " incompatible derivative");
            Matrix block = g.passive_map * response + g.input_map;
            for (int i = 0; i < static_cast<int>(g.indices.size()); ++i)
                for (int k = 0; k < static_cast<int>(g.inputs.size()); ++k)
                    d(g.indices[i], g.inputs[k]) = block(i, k);
        }
        work.derivative = d;
        work.has_derivative = true;
        return work.derivative;
    }
    Inputs motors(const Tree& state) const {
        Inputs x;
        for (int i = 0; i < 12; ++i) x[i] = state[motor_index[i]];
        return x;
    }
    GlovePose pose(const Tree& state, Work& work) const {
        validate(state, work);
        const auto& d = derivatives(state, work);
        const auto& f = cached_frames(state, work);
        GlovePose result;
        Inputs q = angles * state;
        Inputs m = motors(state);
        for (int i = 0; i < 68; ++i) result.tree_rad[i] = state[i];
        for (int i = 0; i < 12; ++i) {
            result.joint_rad[i] = q[i]; result.motor_rad[i] = m[i];
            for (int j = 0; j < 12; ++j) result.motor_jacobian[i*12+j] = d(motor_index[i], j);
        }
        std::array<V3, 21> points;
        std::array<Eigen::Matrix<double, 3, 12>, 21> jac;
        for (int i = 0; i < 21; ++i) {
            points[i] = world_point(sites[i].link, sites[i].local, f);
            jac[i].setZero();
            for (int node = sites[i].link; node >= 0; node = nodes[node].parent_node) {
                int t = nodes[node].tree;
                if (t >= 0) jac[i].noalias() += point_derivative(node, points[i], f) * d.row(t);
            }
        }
        points[1] = (points[0] + points[2]) * 0.5;
        jac[1] = (jac[0] + jac[2]) * 0.5;
        const V3 wrist = points[0];
        const Eigen::Matrix<double, 3, 12> wrist_jac = jac[0];
        for (int i = 0; i < 21; ++i) {
            V3 p = canonical_to_model.transpose() * (points[i] - wrist);
            Eigen::Matrix<double, 3, 12> j = canonical_to_model.transpose() * (jac[i] - wrist_jac);
            for (int k = 0; k < 3; ++k) {
                result.points_m[3*i+k] = p[k];
                for (int c = 0; c < 12; ++c) result.point_jacobian[(3*i+k)*12+c] = j(k,c);
            }
        }
        result.max_gap_m = work.max_gap;
        return result;
    }

    Tree advance(const Tree& previous, const Inputs& requested, const std::vector<int>& active,
                 bool motor_space, int depth, Work& work) const {
        Inputs old = motor_space ? motors(previous) : Inputs(angles * previous);
        Tree predicted = previous;
        if (!active.empty()) {
            Derivative d = derivatives(previous, work);
            if (motor_space) {
                Matrix motor_d(12,12);
                for (int i = 0; i < 12; ++i) motor_d.row(i) = d.row(motor_index[i]);
                Matrix identity = Matrix::Identity(12,12);
                Matrix inverse = least_squares(motor_d, identity, "motor forward derivative");
                predicted += d * inverse * (requested - old);
            } else predicted += d * (requested - old);
            double jump = 0;
            for (int i = 0; i < 68; ++i) if (branch_mask[i]) jump = std::max(jump, std::abs(predicted[i]-previous[i]));
            if (jump > 0.2 && depth < 8) {
                Inputs middle = (old + requested) * 0.5;
                Tree halfway = advance(previous, middle, active, motor_space, depth+1, work);
                return advance(halfway, requested, active, motor_space, depth+1, work);
            }
        }
        Tree candidate = previous;
        bool converged = true;
        double jump = 0;
        for (int gi : active) {
            const auto& g = groups[gi];
            Vector x(g.passive), input(g.inputs.size());
            for (int i = 0; i < g.passive; ++i) x[i] = std::clamp(predicted[g.indices[i]], g.lower[i], g.upper[i]);
            for (int i = 0; i < static_cast<int>(g.inputs.size()); ++i) input[i] = requested[g.inputs[i]];
            bool cached = false;
            Vector last_x, last_r;
            Matrix last_j;
            auto eval = [&](const Vector& xx, Vector& r, Matrix& jac) {
                if (cached && xx.size() == last_x.size() && (xx.array() == last_x.array()).all()) {
                    r = last_r; jac = last_j; return;
                }
                Vector values(g.indices.size());
                if (motor_space) { values.head(g.passive) = xx; values.tail(input.size()) = input; }
                else values = g.passive_map * xx + g.input_map * input;
                Tree local = candidate;
                for (int i = 0; i < static_cast<int>(g.indices.size()); ++i) local[g.indices[i]] = values[i];
                Matrix full;
                fill_frames(local, g.frame_nodes, work.scratch_frames);
                residual_jac(work.scratch_frames, g, r, full);
                if (motor_space) jac = full.leftCols(g.passive);
                else jac = full * g.passive_map;
                last_x = xx; last_r = r; last_j = jac; cached = true;
            };
            const double rtol = std::min(1e-11, tolerance / std::sqrt(3.0));
            Vector r; Matrix j;
            for (int iteration = 0; iteration < 6; ++iteration) {
                eval(x, r, j);
                if (r.cwiseAbs().maxCoeff() <= rtol) break;
                Vector step;
                try { step = least_squares(j, -r, g.name + " correction"); }
                catch (const std::runtime_error&) { break; }
                double alpha = 1.0;
                for (int i = 0; i < g.passive; ++i) {
                    if (step[i] > 0) alpha = std::min(alpha, (g.upper[i]-x[i])/step[i]);
                    else if (step[i] < 0) alpha = std::min(alpha, (g.lower[i]-x[i])/step[i]);
                }
                if (alpha <= 0) break;
                double cost = r.squaredNorm();
                bool accepted = false;
                for (int trial = 0; trial < 8; ++trial) {
                    Vector next = (x + alpha * step).cwiseMax(g.lower).cwiseMin(g.upper);
                    Vector nr; Matrix nj; eval(next, nr, nj);
                    if (nr.squaredNorm() < cost) { x = next; accepted = true; break; }
                    alpha *= 0.5;
                }
                if (!accepted) break;
            }
            eval(x, r, j);
            if (r.cwiseAbs().maxCoeff() > rtol) { converged = false; break; }
            Vector values(g.indices.size());
            if (motor_space) { values.head(g.passive) = x; values.tail(input.size()) = input; }
            else values = g.passive_map * x + g.input_map * input;
            for (int i = 0; i < static_cast<int>(g.indices.size()); ++i) {
                int index = g.indices[i]; candidate[index] = values[i];
                if (branch_mask[index]) jump = std::max(jump, std::abs(values[i]-previous[index]));
            }
        }
        if (converged && jump <= 0.2) {
            try { validate(candidate, work); derivatives(candidate, work); return candidate; }
            catch (const std::runtime_error&) { /* Retry smaller continuation steps. */ }
        }
        if (depth >= 8 || (requested-old).cwiseAbs().maxCoeff() < 1e-5)
            fail("target outside continuous assembly branch or closure did not converge");
        Inputs middle = (old + requested) * 0.5;
        Tree halfway = advance(previous, middle, active, motor_space, depth+1, work);
        return advance(halfway, requested, active, motor_space, depth+1, work);
    }

    Impl(const std::filesystem::path& root, bool left);
    GlovePose evaluate(const Vec12& target, const GlovePose* seed, bool motor_space) const {
        require_finite(target);
        const Vec68& source = seed ? seed->tree_rad : zero_pose.tree_rad;
        require_finite(source);
        Tree state = Eigen::Map<const Tree>(source.data());
        Work work;
        validate(state, work); derivatives(state, work);
        Inputs requested = Eigen::Map<const Inputs>(target.data());
        Inputs old = motor_space ? motors(state) : Inputs(angles * state);
        if ((requested-old).cwiseAbs().maxCoeff() > 4.8) fail("single input change exceeds 4.8 radians");
        std::vector<int> active;
        for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
            for (int input : groups[i].inputs) if (requested[input] != old[input]) { active.push_back(i); break; }
        }
        return pose(advance(state, requested, active, motor_space, 0, work), work);
    }
};

Ve33Model::Impl::Impl(const std::filesystem::path& root, bool left) {
    const auto dir = root / (left ? "hand_motor_ve33_L" : "hand_motor_ve33_R");
    const auto cfg = read_json(dir / "config/joint_space.json");
    const auto targets = read_json(dir / "config/retarget_targets.json");
    if (cfg.at("side") != (left ? "L" : "R") || targets.at("side") != (left ? "L" : "R"))
        fail("asset side mismatch");
    const auto closure = YAML::LoadFile((dir / "config/loop_closures.yaml").string());
    tolerance = closure["closure_tolerance_m"].as<double>();
    if (!std::isfinite(tolerance) || tolerance <= 0) fail("invalid closure tolerance");

    std::vector<std::string> tree_names;
    for (const char* key : {"independent", "dependent"}) {
        const auto list = closure[key];
        if (!list.IsSequence() || list.size() != (std::string(key) == "independent" ? 12u : 56u))
            fail("invalid tree coordinate list");
        for (const auto& item : list) {
            std::string name = item.as<std::string>();
            if (!tree_index.emplace(name, static_cast<int>(tree_names.size())).second) fail("duplicate tree joint");
            tree_names.push_back(name);
        }
    }

    tinyxml2::XMLDocument xml;
    if (xml.LoadFile((dir / "hand_tree_only.urdf").string().c_str()) != tinyxml2::XML_SUCCESS)
        fail("cannot read CAD URDF");
    const auto* robot = xml.FirstChildElement("robot");
    if (!robot) fail("invalid CAD URDF");
    for (auto* e = robot->FirstChildElement("joint"); e; e = e->NextSiblingElement("joint")) {
        Joint j;
        j.name = attr(e, "name");
        std::string kind = attr(e, "type");
        if (kind == "fixed") j.kind = 0;
        else if (kind == "revolute" || kind == "continuous") j.kind = 1;
        else if (kind == "prismatic") j.kind = 2;
        else fail("unsupported URDF joint " + j.name);
        j.parent = attr(e->FirstChildElement("parent"), "link");
        j.child = attr(e->FirstChildElement("child"), "link");
        const auto* origin = e->FirstChildElement("origin");
        V3 xyz = triple(origin ? origin->Attribute("xyz") : nullptr, V3::Zero());
        V3 rpy = triple(origin ? origin->Attribute("rpy") : nullptr, V3::Zero());
        j.origin.translation() = xyz;
        j.origin.linear() = (Eigen::AngleAxisd(rpy.z(), V3::UnitZ()) *
                             Eigen::AngleAxisd(rpy.y(), V3::UnitY()) *
                             Eigen::AngleAxisd(rpy.x(), V3::UnitX())).toRotationMatrix();
        const auto* axis = e->FirstChildElement("axis");
        j.axis = triple(axis ? axis->Attribute("xyz") : nullptr, V3::UnitZ());
        if (j.axis.norm() == 0) fail("zero URDF joint axis");
        j.axis.normalize();
        const auto* limit = e->FirstChildElement("limit");
        if (kind != "continuous" && limit) {
            if (limit->Attribute("lower") && limit->QueryDoubleAttribute("lower", &j.lower) != tinyxml2::XML_SUCCESS)
                fail("invalid lower joint limit");
            if (limit->Attribute("upper") && limit->QueryDoubleAttribute("upper", &j.upper) != tinyxml2::XML_SUCCESS)
                fail("invalid upper joint limit");
            if (std::isnan(j.lower) || std::isnan(j.upper) || j.lower > j.upper) fail("invalid joint limits");
        }
        auto it = tree_index.find(j.name);
        if (it != tree_index.end()) {
            if (!j.kind) fail("fixed joint in tree coordinates");
            j.tree = it->second;
        } else if (j.kind) fail("unlisted moving URDF joint");
        int index = static_cast<int>(nodes.size());
        if (!joint_node.emplace(j.name, index).second || !child_node.emplace(j.child, index).second)
            fail("duplicate URDF joint or child");
        nodes.push_back(std::move(j));
    }
    for (auto& j : nodes) {
        auto it = child_node.find(j.parent);
        if (it == child_node.end()) roots.insert(j.parent);
        else j.parent_node = it->second;
    }
    std::vector<unsigned char> visited(nodes.size(), 0);
    auto visit = [&](auto&& self, int i) -> void {
        if (visited[i] == 2) return;
        if (visited[i] == 1) fail("URDF cycle");
        visited[i] = 1;
        if (nodes[i].parent_node >= 0) self(self, nodes[i].parent_node);
        visited[i] = 2;
        topology.push_back(i);
    };
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) visit(visit, i);
    for (int i = 0; i < 68; ++i) {
        auto it = joint_node.find(tree_names[i]);
        if (it == joint_node.end()) fail("tree joint missing from URDF");
        tree_node[i] = it->second;
    }
    Tree zero = Tree::Zero();
    const auto zero_frames = frames(zero);
    const auto& defs = cfg.at("independent");
    const std::array<std::string, 12> names = {
        "thumb_cmc_flex_joint", "thumb_cmc_abd_joint", "thumb_mcp_joint", "thumb_ip_joint",
        "index_mcp_flex_joint", "index_pip_joint", "middle_mcp_flex_joint", "middle_pip_joint",
        "ring_mcp_flex_joint", "ring_pip_joint", "pinky_mcp_flex_joint", "pinky_pip_joint"};
    if (!defs.is_array() || defs.size() != 12) fail("expected twelve joint definitions");
    std::set<int> unique_motors;
    for (int i = 0; i < 12; ++i) {
        const auto& def = defs[i];
        if (def.at("name").get<std::string>() != names[i]) fail("joint definition order mismatch");
        const std::string motor_name = def.at("motor").get<std::string>();
        auto motor = tree_index.find(motor_name);
        if (motor == tree_index.end() || motor->second >= 12 || !unique_motors.insert(motor->second).second)
            fail("invalid motor definitions");
        motor_index[i] = motor->second;
        int parent = link(def.at("parent").get<std::string>());
        int child = link(def.at("child").get<std::string>());
        auto axis_it = joint_node.find(def.at("axis_joint").get<std::string>());
        if (axis_it == joint_node.end() || child < 0) fail("invalid angle reference");
        V3 axis = world_axis(axis_it->second, zero_frames).normalized();
        double alignment = axis.dot(zero_frames[child].linear().col(2));
        if (std::abs(alignment) < 1e-8) fail("ambiguous physical angle axis");
        if (alignment < 0) axis = -axis;
        for (int n = 0; n < static_cast<int>(nodes.size()); ++n) {
            bool on_parent = ancestor(parent, n), on_child = ancestor(child, n);
            if (on_parent == on_child || nodes[n].kind == 0) continue;
            if (nodes[n].kind != 1 || nodes[n].tree < 0) fail("invalid physical angle path");
            double dot = axis.dot(world_axis(n, zero_frames).normalized());
            if (std::abs(std::abs(dot) - 1.0) > 1e-8) fail("non-collinear physical angle path");
            angles(i, nodes[n].tree) = (on_child ? 1.0 : -1.0) * (dot > 0 ? 1.0 : -1.0);
        }
    }

    for (const auto& c : closure["closures"]) {
        int a = link(c["link_a"].as<std::string>()), b = link(c["link_b"].as<std::string>());
        V3 p = triple(c["point"]), axis = triple(c["axis"]);
        if (axis.norm() == 0) fail("zero closure axis");
        axis.normalize();
        for (int i = 0; i < 2; ++i) {
            V3 world = p + (i * 0.03) * axis;
            witnesses.push_back({a, b, a < 0 ? world : V3(zero_frames[a].inverse() * world),
                                 b < 0 ? world : V3(zero_frames[b].inverse() * world)});
        }
    }
    if (witnesses.empty()) fail("no closure witnesses");
    branch_mask.fill(true);
    std::set<int> covered;
    for (const auto& entry : closure["groups"]) {
        Group g;
        g.name = entry.first.as<std::string>();
        const auto passive = entry.second;
        if (!passive.IsSequence() || passive.size() == 0) fail("empty closure group");
        for (const auto& item : passive) {
            auto it = tree_index.find(item.as<std::string>());
            if (it == tree_index.end() || it->second < 12 || !covered.insert(it->second).second)
                fail("invalid passive group");
            g.indices.push_back(it->second);
        }
        g.passive = static_cast<int>(g.indices.size());
        for (int i = 0; i < 12; ++i) {
            const auto& name = tree_names[motor_index[i]];
            if (name.substr(0, name.find('_')) == g.name) {
                g.inputs.push_back(i);
                if (!covered.insert(motor_index[i]).second) fail("duplicate group motor");
                g.indices.push_back(motor_index[i]);
            }
        }
        const int count = static_cast<int>(g.inputs.size());
        if (!count) fail("group has no motors");
        Matrix a(count, g.indices.size());
        for (int r = 0; r < count; ++r) {
            for (int c = 0; c < 68; ++c)
                if (std::find(g.indices.begin(), g.indices.end(), c) == g.indices.end() && angles(g.inputs[r],c) != 0)
                    fail("physical joint crosses closure groups");
            for (int c = 0; c < static_cast<int>(g.indices.size()); ++c)
                a(r,c) = angles(g.inputs[r],g.indices[c]);
        }
        Matrix drive = a.rightCols(count);
        Eigen::FullPivLU<Matrix> lu(drive);
        if (!lu.isInvertible()) fail("physical angle motor map singular");
        Matrix inverse = lu.inverse();
        g.passive_map = Matrix::Zero(g.indices.size(), g.passive);
        g.passive_map.topRows(g.passive).setIdentity();
        g.passive_map.bottomRows(count) = -inverse * a.leftCols(g.passive);
        g.input_map = Matrix::Zero(g.indices.size(), count);
        g.input_map.bottomRows(count) = inverse;
        g.lower.resize(g.passive); g.upper.resize(g.passive);
        for (int i = 0; i < g.passive; ++i) {
            const auto& n = nodes[tree_node[g.indices[i]]];
            g.lower[i] = n.lower; g.upper[i] = n.upper;
        }
        for (int w = 0; w < static_cast<int>(witnesses.size()); ++w) {
            for (int i = 0; i < g.passive; ++i) {
                int n = tree_node[g.indices[i]];
                if (ancestor(witnesses[w].a, n) || ancestor(witnesses[w].b, n)) {
                    g.witnesses.push_back(w); break;
                }
            }
        }
        if (g.witnesses.empty()) fail("group has no closure witnesses");
        std::vector<unsigned char> required(nodes.size(), 0);
        for (int wi : g.witnesses) {
            const auto& w = witnesses[wi];
            for (int n = w.a; n >= 0; n = nodes[n].parent_node) required[n] = 1;
            for (int n = w.b; n >= 0; n = nodes[n].parent_node) required[n] = 1;
            g.mask_a.emplace_back(g.indices.size());
            g.mask_b.emplace_back(g.indices.size());
            for (int i = 0; i < static_cast<int>(g.indices.size()); ++i) {
                int n = tree_node[g.indices[i]];
                g.mask_a.back()[i] = ancestor(w.a, n);
                g.mask_b.back()[i] = ancestor(w.b, n);
            }
        }
        for (int n : topology) if (required[n]) g.frame_nodes.push_back(n);
        for (int c = 0; c < count; ++c) {
            int driven = -1;
            for (int i = 0; i < static_cast<int>(g.indices.size()); ++i) {
                if (std::abs(g.input_map(i,c)) > 1e-12) {
                    if (driven >= 0) { driven = -2; break; }
                    driven = i;
                }
            }
            if (driven < 0 || g.passive_map.row(driven).cwiseAbs().maxCoeff() > 1e-12) continue;
            int n = tree_node[g.indices[driven]];
            bool rigid = true;
            for (int wi : g.witnesses)
                if (ancestor(witnesses[wi].a,n) != ancestor(witnesses[wi].b,n)) { rigid = false; break; }
            if (rigid) branch_mask[g.indices[driven]] = false;
        }
        groups.push_back(std::move(g));
    }
    if (covered.size() != 68) fail("closure groups do not cover tree");

    const auto& rotation = targets.at("canonical_to_model");
    if (!rotation.is_array() || rotation.size() != 3) fail("invalid canonical rotation");
    for (int r = 0; r < 3; ++r) {
        if (!rotation[r].is_array() || rotation[r].size() != 3) fail("invalid canonical rotation");
        for (int c = 0; c < 3; ++c) canonical_to_model(r,c) = rotation[r][c].get<double>();
    }
    if (!canonical_to_model.allFinite() ||
        (canonical_to_model.transpose()*canonical_to_model - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-6 ||
        std::abs(canonical_to_model.determinant() - 1.0) > 1e-6) fail("canonical rotation is not proper");
    const auto& list = targets.at("sites");
    if (!list.is_array() || list.size() != 21) fail("expected 21 landmarks");
    std::array<bool,21> seen{};
    for (const auto& item : list) {
        int i = item.at("index").get<int>();
        if (i < 0 || i >= 21 || seen[i]) fail("invalid landmark index");
        seen[i] = true;
        sites[i] = {link(item.at("link").get<std::string>()), triple(item.at("local_xyz_m"))};
    }
    Work work;
    zero_pose = pose(zero, work);
}

Ve33Model::Ve33Model(const std::filesystem::path& root, bool left)
    : impl_(std::make_unique<Impl>(root, left)) {}
Ve33Model::~Ve33Model() = default;
Ve33Model::Ve33Model(Ve33Model&&) noexcept = default;
Ve33Model& Ve33Model::operator=(Ve33Model&&) noexcept = default;

GlovePose Ve33Model::zero() { return impl_->zero_pose; }
GlovePose Ve33Model::evaluate_motors(const Vec12& motors, const GlovePose* seed) {
    return impl_->evaluate(motors, seed, true);
}
GlovePose Ve33Model::evaluate_joints(const Vec12& joints, const GlovePose* seed) {
    return impl_->evaluate(joints, seed, false);
}
GlovePose Ve33Model::evaluate_tree(const Vec68& tree) {
    require_finite(tree);
    Impl::Work work;
    return impl_->pose(Eigen::Map<const Tree>(tree.data()), work);
}
std::array<std::array<double, 2>, 12> Ve33Model::active_joint_bounds(const GlovePose& pose) {
    require_finite(pose.tree_rad);
    Tree state = Eigen::Map<const Tree>(pose.tree_rad.data());
    Impl::Work work;
    impl_->validate(state, work);
    const Derivative& d = impl_->derivatives(state, work);
    Inputs q = impl_->angles * state;
    std::array<std::array<double, 2>, 12> bounds;
    for (auto& b : bounds) b = {-inf, inf};
    for (int i = 0; i < 68; ++i) {
        const auto& node = impl_->nodes[impl_->tree_node[i]];
        double magnitude = d.row(i).cwiseAbs().maxCoeff();
        if (magnitude == 0) continue;
        int joint = -1;
        for (int k = 0; k < 12; ++k) if (std::abs(d(i,k)) > magnitude * 1e-6) {
            if (joint >= 0) { joint = -2; break; }
            joint = k;
        }
        if (joint < 0) continue;
        double slope = d(i,joint);
        for (int side = 0; side < 2; ++side) {
            double bound = side == 0 ? node.lower : node.upper;
            if (std::abs(bound-state[i]) > 1e-7) continue;
            double value = q[joint] + (bound-state[i])/slope;
            if ((slope > 0) == (side == 0)) bounds[joint][0] = std::max(bounds[joint][0], value);
            else bounds[joint][1] = std::min(bounds[joint][1], value);
        }
    }
    return bounds;
}

}  // namespace motorhand::native
