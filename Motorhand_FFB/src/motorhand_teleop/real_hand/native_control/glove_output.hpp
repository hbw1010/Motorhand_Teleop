#pragma once
#include "common/glove_output.hpp"
#include "jacobian_worker.hpp"
#include "hands/wuji/compat.hpp"
namespace motorhand::native {
using GloveOutputResult=CommonGloveOutputResult<20>;
class GloveOutputWorker : public CommonGloveOutputWorker<20> {
public:
    using CommonGloveOutputWorker<20>::CommonGloveOutputWorker;
    using CommonGloveOutputWorker<20>::submit_force;
    std::uint64_t submit_force(const Vec20& t,const HandFrame& h,const GloveFrame& g,const GlovePose& p,
        const std::optional<CommonJacobianSnapshot<20>>& j,std::uint64_t deadline) {
        return CommonGloveOutputWorker<20>::submit_force(t,common_wuji_frame(h),g,p,j,deadline);
    }
};
}
