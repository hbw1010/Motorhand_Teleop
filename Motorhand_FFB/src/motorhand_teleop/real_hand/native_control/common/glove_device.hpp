#pragma once
#include "encos_driver.hpp"
#include <memory>
namespace motorhand::native {
// 共同会话和唯一输出线程所需的12轴接口；原始收包时间由设备拥有。
class GloveDevice {
public:
    virtual ~GloveDevice()=default;
    virtual void connect()=0;
    virtual GloveFrame read()=0;
    virtual std::array<EncosMotorCalibration,12> calibration() const=0;
    virtual std::array<std::uint64_t,12> last_command_write_ns() const=0;
    virtual void home()=0;
    virtual void enable_force()=0;
    virtual Vec12 write_position(const Vec12&,std::uint64_t original_deadline_ns)=0;
    virtual Vec12 write_torque(const Vec12&,std::uint64_t original_deadline_ns)=0;
    virtual DriverImpedanceSubmission write_impedance(const DriverImpedanceCommand&,std::uint64_t original_deadline_ns)=0;
    virtual void disable()=0;
    virtual void close()=0;
};
// 保留原EncosDriver的ABI；会话拥有驱动，旧输出工作器构造只借用驱动。
class EncosGlove final : public GloveDevice {
public:
    explicit EncosGlove(EncosDriver& driver):driver_(&driver){}
    EncosGlove(const std::string& mapping,const std::string& limits)
        :owned_(std::make_unique<EncosDriver>(mapping,limits)),driver_(owned_.get()){}
    void connect() override{driver_->connect();}
    GloveFrame read() override{return driver_->read();}
    std::array<EncosMotorCalibration,12> calibration() const override{return driver_->calibration();}
    std::array<std::uint64_t,12> last_command_write_ns() const override{return driver_->last_command_write_ns();}
    void home() override{driver_->home();}
    void enable_force() override{driver_->enable_force();}
    Vec12 write_position(const Vec12& v,std::uint64_t until) override{return driver_->write_position(v,until);}
    Vec12 write_torque(const Vec12& v,std::uint64_t until) override{return driver_->write_torque(v,until);}
    DriverImpedanceSubmission write_impedance(const DriverImpedanceCommand& v,std::uint64_t until) override{return driver_->write_impedance(v,until);}
    void disable() override{driver_->disable();}
    void close() override{driver_->close();}
private:
    std::unique_ptr<EncosDriver> owned_;
    EncosDriver* driver_;
};
}
