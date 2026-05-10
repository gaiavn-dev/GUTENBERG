// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>

#include <cuda_runtime.h>

namespace chrono = std::chrono;


class CudaStoreTimer
{
    typedef decltype(chrono::high_resolution_clock::now()) tp_t;
public:
    CudaStoreTimer(double &storage, bool enabled=true, bool synchronize=true)
        : m_storage(&storage), m_enabled(enabled), m_synchronize(synchronize)
    {
        m_start = GetTP();
    }
    ~CudaStoreTimer()
    {
        if (! m_enabled) return;

        auto tNow = GetTP();
        chrono::duration<double, std::milli> dur = tNow - m_start;
        *m_storage = dur.count();
    }

private:
    tp_t GetTP() const
    {
        if (m_enabled && m_synchronize) {
            cudaDeviceSynchronize();
        }
        return chrono::high_resolution_clock::now();
    }

    double *m_storage;
    tp_t m_start;
    bool m_enabled;
    bool m_synchronize;
};
