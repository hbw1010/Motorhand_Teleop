#pragma once
#include "common/hand.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace motorhand::native::simulation {
inline constexpr const char* sharpa_contact_load_source="sharpa-mujoco-contact-joint-torque";
struct SharpaSnapshot {
    JointArray<22> position_rad{},target_rad{},contact_load_nm{};
    double simulation_time{};
    int contact_count{};
    std::vector<double> qpos;
    std::string scene_xml;
};
// 本实例独占模型/数据与500Hz动力学线程；read/snapshot只复制缓存。
class SharpaDevice final : public HandDevice<22> {
public:
    SharpaDevice(std::filesystem::path asset_root,HandDescriptor<22> descriptor,bool with_object=true);
    ~SharpaDevice();
    SharpaDevice(const SharpaDevice&)=delete;
    SharpaDevice& operator=(const SharpaDevice&)=delete;
    void connect() override;
    HandState<22> read() override;
    const HandDescriptor<22>& descriptor() const override;
    void enable() override;
    void home(const std::atomic<bool>* stop_requested=nullptr) override;
    void send(const JointArray<22>&,std::uint64_t original_deadline_ns) override;
    void disable() override;
    void close() override;
    SharpaSnapshot snapshot() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace motorhand::native::simulation
