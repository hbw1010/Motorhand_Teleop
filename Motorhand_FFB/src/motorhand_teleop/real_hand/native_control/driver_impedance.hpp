#pragma once
#include "common/driver_impedance.hpp"
#include "hands/wuji/compat.hpp"
namespace motorhand::native {
using DriverImpedancePlan=CommonDriverImpedancePlan<20>;
class DriverImpedancePlanner : public CommonDriverImpedancePlanner<20> {
public:
    using CommonDriverImpedancePlanner<20>::CommonDriverImpedancePlanner;
    using CommonDriverImpedancePlanner<20>::step;
    DriverImpedancePlan step(const Vec20& t,const HandFrame& h,const GloveFrame& g,const GlovePose& p,
        const RetargetJacobian<20>& j,const ForceResult& f,std::uint64_t n) {
        return CommonDriverImpedancePlanner<20>::step(t,common_wuji_frame(h),g,p,j,common_wuji_feedback(f),n);
    }
};
}
