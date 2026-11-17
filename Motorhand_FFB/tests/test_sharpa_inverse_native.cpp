// 真实MuJoCo首帧的微小超调曾使零电机限位残差触发SLSQP反复roundoff失败。
#include "common/inverse_retarget.hpp"
#include "hands/sharpa/model.hpp"
#include "ve33_model.hpp"
#include "encos_driver.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
using namespace motorhand::native;
int main(int argc,char** argv){try{
    const auto root=argc>1?std::filesystem::path(argv[1]):std::filesystem::current_path();
    Ve33Model glove(root/"assets/hand_motor_ve33",false);
    sharpa::Model hand(root/"Motorhand_FFB/assets/sharpa-hand",false);
    // 只读本地标定元数据；不connect任何真实设备。
    EncosDriver metadata((root/"hardware/encos_dexhand/configs/sim/glove_mapping.json").string());
    const auto cal=metadata.calibration();std::array<std::array<double,2>,12> bounds{};
    for(int i=0;i<12;++i)bounds[i]={cal[i].model_min_rad,cal[i].model_max_rad};
    const auto zero=glove.zero();
    const JointArray<22> measured={3.0788083443515223e-09,-7.580783990157879e-06,-9.400578659612215e-07,-5.394254973173509e-07,-3.1742723486461034e-07,3.064992414154025e-09,2.240168583138159e-08,-2.1177682149355153e-07,-1.1138036691935847e-07,3.064992414154163e-09,2.240168583138132e-08,-2.1177682149355222e-07,-1.1138036691935875e-07,3.0649924141540377e-09,2.2401685831381125e-08,-2.1177682149355222e-07,-1.1138036691935875e-07,-7.929232198664313e-11,3.0815223720880726e-09,2.2444141595400738e-08,-2.117726690279212e-07,-1.1137831519296002e-07};
    for(const auto& q:{JointArray<22>{},measured,hand.solve(zero.points_m)}) {
        CommonInverseRetarget<22> inverse(glove,hand,bounds);
        const auto result=inverse.solve(q,zero);
        if(!result.success)throw std::runtime_error(result.failure_reason);
        if(!std::isfinite(result.task_error_rms_mm))throw std::runtime_error("逆解误差非有限");
        for(int i=0;i<12;++i)if(result.pose.motor_rad[i]<bounds[i][0]||result.pose.motor_rad[i]>bounds[i][1])
            throw std::runtime_error("最终电机目标必须严格满足原始标定限位");
    }
    std::cout<<"Sharpa零位、真实首帧与前向零位的共同逆解通过\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
