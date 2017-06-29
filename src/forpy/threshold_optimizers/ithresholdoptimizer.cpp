#include <forpy/threshold_optimizers/ithresholdoptimizer.h>

namespace forpy {
  IThresholdOptimizer::IThresholdOptimizer() {};
  IThresholdOptimizer::~IThresholdOptimizer() {};

  FORPY_IMPL_DIRECT(FORPY_ITHRESHOPT_EARLYSTOP,\
                    ,\
                    AT,\
                    IThresholdOptimizer,\
                    { return false; };)

  void IThresholdOptimizer::prepare_for_optimizing(const size_t &node_id,
                                                   const int &num_threads) {};

  FORPY_IMPL_NOTAVAIL(FORPY_ITHRESHOPT_OPT, ITFTAT, IThresholdOptimizer);
} // namespace IThresholdOptimizer

