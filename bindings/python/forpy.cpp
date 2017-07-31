#include "./forpy_exporters.h"
#include <forpy/global.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace forpy;

PYBIND11_MODULE(forpy, m) {
    m.doc() = "Forpy python interface.";

    export_global(m);
    export_types(m);
    export_impurities(m);
    export_gains(m);
    export_regressors(m);
    export_threshold_optimizers(m);
    export_util(m);
    export_data_providers(m);
    export_feature(m);
    export_leafs(m);
    export_deciders(m);
    export_tree(m);

    forpy::init();
}
