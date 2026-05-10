// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "quad_rectify/quad_rectify.h"
#include "non_maximal_suppression/non_maximal_suppression.h"
#include "geometry_api/geometry_api.h"
#include "beam_decode/beam_decode.h"
#include "better_grid_sample/grid_sample.h"
#include "sparse_select/sparse_select.h"
#include "text_region_grouping/text_region_grouping.h"
#include "local_ips/local_ips.h"

#include <torch/extension.h>
#include <pybind11/stl.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("quad_rectify_calc_quad_width", &quad_rectify_calc_quad_width,
            "Quad Rectify Calc Quad Width C++",
            py::arg("quads"),
            py::arg("output_height"),
            py::arg("round_factor") = 16,
            py::arg("max_width") = 0
      );
      m.def("quad_rectify_forward", &quad_rectify_forward, "Quad Rectify Forward C++",
            py::arg("quads"),
            py::arg("image_height"), py::arg("image_width"),
            py::arg("output_height"), py::arg("output_width"),
            py::arg("isotropic") = true
      );
      m.def("quad_rectify_backward", &quad_rectify_backward, "Quad Rectify Backward C++",
            py::arg("quads"), py::arg("grad_output"),
            py::arg("image_height"), py::arg("image_width"),
            py::arg("isotropic") = true
      );
      m.def("quad_non_maximal_suppression", &quad_non_maximal_suppression, "Quad Non-Maximal Suppression C++",
            py::arg("quads"), py::arg("probs"),
            py::arg("prob_threshold"), py::arg("iou_threshold"),
            py::arg("kernel_height"), py::arg("kernel_width"),
            py::arg("max_regions"),
            py::arg("verbose") = false
      );

      py::class_<LanguageModel>(m, "LanguageModel");

      m.def("beam_decode", &beam_decode, "beam_decode c++",
            py::arg("probs"),
            py::arg("beam_size") = 100,
            py::arg("blank") = 0,
            py::arg("min_prob") = 0.001,
            py::arg("lang_model") = static_cast<LanguageModel*>(nullptr),
            py::arg("lm_weight") = 1,
            py::arg("combine_duplicates") = true
      );

      py::class_<TokenMappingWrapper, TokenMappingWrapper::Ptr>(m, "TokenMapping");

      m.def("create_token_mapping", &create_token_mapping, "create token mapping c++",
            py::arg("token_mapping")
      );

      m.def("decode_sequences", &decode_sequences, "decode_sequences c++",
            py::arg("tokens"), py::arg("language_model"),
            py::arg("probs") = nullptr
      );

      m.def("create_sbo_lm", &create_sbo_lm, "create_sbo_lm c++",
            py::arg("data_file_path"),
            py::arg("token_mapping"),
            py::arg("backoff") = 0.4
      );

      m.def("indirect_grid_sample_forward", &indirect_grid_sample_forward, "indirect_grid_sample::forward c++",
            py::arg("input"), py::arg("grid"), py::arg("input_indices"), py::arg("method")
      );
      m.def("indirect_grad_sample_backward", &indirect_grad_sample_backward, "indirect_grid_sample::backward c++",
            py::arg("grad_output"), py::arg("input"), py::arg("grid"), py::arg("input_indices"), py::arg("method")
      );
      m.def("region_counts_to_indices", &region_counts_to_indices, "region counts to indices",
            py::arg("region_counts"), py::arg("num_outputs")
      );

      m.def("rrect_to_quads", &rrect_to_quads, "convert rotated rectangle to quadrangles",
            py::arg("rrects"), py::arg("cell_size")
      );
      m.def("rrect_to_quads_backward", &rrect_to_quads_backward, "gradient of rrect_to_quads",
            py::arg("rrects"), py::arg("grad_output")
      );

      m.def("sparse_select", &sparse_select, "Select sparse tensor(s) given a set of indices",
            py::arg("sparse_counts"), py::arg("sparse_tensors"), py::arg("select_indices")
      );

      m.def("text_region_grouping", &text_region_grouping, "Clusters all of the text into lines and phrases",
            py::arg("quads"), py::arg("counts"),
            py::arg("horizontal_tolerance") = 2.0f,
            py::arg("vertical_tolerance") = 0.5f,
            py::arg("verbose") = false
      );

      m.def("dense_relations_to_graph", &dense_relations_to_graph, "Converts a dense relational tensor to a graph",
            py::arg("relations")
      );

      m.def("ragged_quad_all_2_all_distance_v2", &ragged_quad_all_2_all_distance_v2, "get the all-to-all distances in ragged-batch quad mode",
            py::arg("embed_quads"), py::arg("region_counts"),
            py::arg("x_factor") = 1.0f,
            py::arg("y_factor") = 1.0f,
            py::arg("allow_self_distance") = true
      );

      m.def("calc_poly_min_rrect", &calc_poly_min_rrect, "calculate a reasonable bounding rectangle for a given text polygon",
            py::arg("vertices")
      );

      m.def("get_rel_continuation_cos", &get_rel_continuation_cos, "c++ get relation cosine between 2 regions",
            py::arg("rrect_a"), py::arg("rrect_b")
      );

      m.def("get_poly_bounds_quad", &get_poly_bounds_quad, "c++ get polygon bounds",
            py::arg("poly")
      );
}
