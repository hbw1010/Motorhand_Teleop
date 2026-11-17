#pragma once
#include "common/jacobian_worker.hpp"
namespace motorhand::native {
using JacobianWorker = CommonJacobianWorker<20>;
using JacobianSnapshot = CommonJacobianSnapshot<20>;
}
