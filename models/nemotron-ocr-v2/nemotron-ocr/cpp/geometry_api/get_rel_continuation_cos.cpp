// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_api.h"

#include "../geometry.h"

using namespace std;


float get_rel_continuation_cos(torch::Tensor rrectATensor, torch::Tensor rrectBTensor)
{
    typedef Point_<float> Pointf;

    if (rrectATensor.size(0) != 4 || rrectBTensor.size(0) != 4) {
        throw runtime_error("Invalid rrect arguments. Both must have 4 vertices! A=" +
                                to_string(rrectATensor.size(0)) + ", B=" + to_string(rrectBTensor.size(0)));
    }

    auto rrectA = rrectATensor.accessor<float, 2>();
    auto rrectB = rrectBTensor.accessor<float, 2>();

    Pointf aPts[4] = {
        rrectA[0], rrectA[1], rrectA[2], rrectA[3]
    };

    auto c1 = (aPts[0] + aPts[3]) / 2.0f;
    auto c2 = (aPts[1] + aPts[2]) / 2.0f;

    auto aDir = c2 - c1;
    auto aLen = length(aDir);

    if (aLen > 0) {
        aDir /= aLen;
    } else {
        aDir = Pointf{ 1, 0 };
    }

    auto centerA = (c1 + c2) / 2.0f;

    Pointf bPts[4] = {
        rrectB[0], rrectB[1], rrectB[2], rrectB[3]
    };

    auto centerB = (bPts[0] + bPts[1] + bPts[2] + bPts[3]) / 4.0f;

    auto connDir = centerB - centerA;
    auto connLen = length(connDir);

    if (connLen == 0.0f) {
        return 1.0f;
    }

    connDir /= connLen;

    auto cosT = dot(aDir, connDir);

    return cosT;
}
