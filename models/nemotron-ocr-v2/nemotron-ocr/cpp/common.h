// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>
#include <vector>

#include <torch/torch.h>

template<typename T>
inline
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[";
    if (! v.empty()) {
        os << v[0];
        for (size_t i = 1; i < v.size(); ++i) {
            os << ", " << v[i];
        }
    }
    os << "]";
    return os;
}

template<int Counter, typename ...Args>
struct _inner_tuple_print
{
    inline
    static std::ostream &print(std::ostream &os, const std::tuple<Args...> &t) {
        _inner_tuple_print<Counter - 1, Args...>::print(os, t);

        os << ", " << std::get<Counter>(t);
        return os;
    }
};

template<typename ...Args>
struct _inner_tuple_print<0, Args...>
{
    inline
    static std::ostream &print(std::ostream &os, const std::tuple<Args...> &t) {
        os << std::get<0>(t);
        return os;
    }
};


template<typename... Args>
inline
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t) {
    os << "(";
    _inner_tuple_print<sizeof...(Args) - 1, Args...>::print(os, t);
    os << ")";
    return os;
}

void print_tensor(const torch::Tensor &t);
