#pragma once
#include "common/hand.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>
namespace motorhand::native::sharpa {
enum class MotionTargetSource { measured, submitted };

class MotionRecorder {
public:
    MotionRecorder(const std::filesystem::path& path,const HandDescriptor<22>& d,
                   MotionTargetSource target_source = MotionTargetSource::measured)
        :lower_(d.lower_rad),upper_(d.upper_rad),target_source_(target_source){
        if(std::filesystem::exists(path))throw std::runtime_error("轨迹文件已存在，拒绝覆盖");
        if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
        out_.open(path);if(!out_)throw std::runtime_error("无法创建轨迹文件");
        out_<<nlohmann::json{{"schema_version",2},{"robot","sharpa"},{"hand",d.identity.left?"left":"right"},
            {"serial",d.identity.serial},{"joint_names",d.joint_names},{"position_unit","rad"},
            {"load_unit","Nm"},{"load_source",d.load_source},
            {"target_source",target_source==MotionTargetSource::submitted
                ? "submitted_position_target" : "measured_position_projected_to_command_limits"}}.dump()<<'\n';
    }
    void append(std::uint64_t source_ns,const JointArray<22>& q){
        append_frame(source_ns,q,nlohmann::json::object());
    }
    void append(const HandState<22>& s){
        append_state(s, nullptr);
    }
    void append(const HandState<22>& s,const JointArray<22>& submitted){
        append_state(s, &submitted);
    }
    void finish(){out_.flush();if(!out_)throw std::runtime_error("轨迹写入未完成");out_.close();if(!out_)throw std::runtime_error("轨迹关闭失败");}
private:
    void append_state(const HandState<22>& s,const JointArray<22>* submitted){
        append_frame(s.received_ns,s.position_rad,{{"source_ns",s.received_ns},{"device_time",s.device_time},
            {"sequence",s.sequence},{"velocity_rad_s",s.has_velocity?nlohmann::json(s.velocity_rad_s):nlohmann::json(nullptr)},
            {"load_nm",s.has_load?nlohmann::json(s.load):nlohmann::json(nullptr)},{"load_source",s.load_source}},submitted);
    }
    void append_frame(std::uint64_t source_ns,const JointArray<22>& q,nlohmann::json row,
                      const JointArray<22>* submitted=nullptr){
        if((submitted!=nullptr)!=(target_source_==MotionTargetSource::submitted))
            throw std::invalid_argument("轨迹目标来源与记录模式不一致");
        if(!source_ns)throw std::invalid_argument("轨迹源时间无效");
        for(double v:q)if(!std::isfinite(v))throw std::invalid_argument("轨迹角度非有限");
        JointArray<22> target{};
        for(int i=0;i<22;++i) {
            target[i]=submitted ? (*submitted)[i] : std::clamp(q[i],lower_[i],upper_[i]);
            if(!std::isfinite(target[i])||target[i]<lower_[i]||target[i]>upper_[i])
                throw std::invalid_argument("已提交的轨迹目标非法");
        }
        if(first_&&source_ns<=last_)return;
        if(!first_)first_=source_ns;
        row["t_ns"]=source_ns-first_;row["position_rad"]=q;row["target_rad"]=target;
        out_<<row.dump()<<'\n';
        if(!out_)throw std::runtime_error("轨迹写入失败");last_=source_ns;
    }
    std::ofstream out_;std::uint64_t first_{},last_{};JointArray<22>lower_{},upper_{};
    MotionTargetSource target_source_;
};
class Motion {
public:
    static Motion load(const std::filesystem::path& path,const HandDescriptor<22>& d){
        std::ifstream f(path);if(!f)throw std::runtime_error("无法读取Sharpa轨迹");
        std::string line;if(!std::getline(f,line))throw std::runtime_error("轨迹为空");
        const auto h=nlohmann::json::parse(line);
        if(!h.at("joint_names").is_array()||h.at("joint_names").size()!=22||h.at("schema_version")!=2||h.at("robot")!="sharpa"||h.at("hand")!=(d.identity.left?"left":"right")||
           h.at("position_unit")!="rad"||h.at("joint_names").get<std::array<std::string,22>>()!=d.joint_names)
            throw std::runtime_error("轨迹型号、手侧、单位或22关节顺序不匹配");
        const std::string serial=h.value("serial","");
        if(!serial.empty()&&d.identity.serial.rfind("SIM-",0)!=0&&serial!=d.identity.serial)
            throw std::runtime_error("轨迹设备身份不匹配");
        Motion result;
        while(std::getline(f,line)){
            if(line.empty())continue;const auto row=nlohmann::json::parse(line);
            if(!row.at("t_ns").is_number_unsigned())throw std::runtime_error("轨迹时间必须为非负整数");
            auto t=row.at("t_ns").get<std::uint64_t>();
            const auto& values=row.contains("target_rad")?row.at("target_rad"):row.at("position_rad");
            if(!values.is_array()||values.size()!=22)throw std::runtime_error("轨迹必须恰有22个目标");
            auto q=values.get<JointArray<22>>();
            if((result.times_.empty()&&t!=0)||(!result.times_.empty()&&t<=result.times_.back()))throw std::runtime_error("轨迹时间必须从0严格递增");
            for(int i=0;i<22;++i)if(!std::isfinite(q[i])||q[i]<d.lower_rad[i]||q[i]>d.upper_rad[i])throw std::runtime_error("轨迹角度超出范围");
            result.times_.push_back(t);result.frames_.push_back(q);
        }
        if(result.frames_.size()<2)throw std::runtime_error("轨迹至少需要两帧");return result;
    }
    std::uint64_t duration_ns()const{return times_.back();}
    JointArray<22> at(std::uint64_t t)const{
        if(t>=times_.back())return frames_.back();
        auto hi=std::upper_bound(times_.begin(),times_.end(),t)-times_.begin();
        if(!hi)return frames_.front();const auto lo=hi-1;
        const double a=double(t-times_[lo])/double(times_[hi]-times_[lo]);JointArray<22> q{};
        for(int i=0;i<22;++i)q[i]=frames_[lo][i]+a*(frames_[hi][i]-frames_[lo][i]);return q;
    }
private:
    std::vector<std::uint64_t>times_;std::vector<JointArray<22>>frames_;
};
}
