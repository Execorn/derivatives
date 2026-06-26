#include <torch/extension.h>
#include "grey_bergomi.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mittag_leffler_cuda", &mittag_leffler_cuda, 
          "Mittag-Leffler evaluation kernel on GPU (double precision)",
          py::arg("z"), py::arg("beta"), py::arg("max_iter") = 500, py::arg("tol") = 1e-9);
    m.def("generate_grey_paths_cuda", &generate_grey_paths_cuda, 
          "Grey Rough Bergomi path simulator on GPU (double precision)",
          py::arg("params"), py::arg("steps"), py::arg("paths"), py::arg("T"), py::arg("dt"));
}
