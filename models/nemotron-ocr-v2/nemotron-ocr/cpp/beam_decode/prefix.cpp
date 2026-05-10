// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "prefix.h"

using namespace std;

vector<token_t> Prefix::ToList() const
{
    vector<token_t> ret;

    auto curr = this;

    while (curr) {
        if (curr->Token != 0) {
            ret.push_back(curr->Token);
        }
        curr = curr->Parent;
    }

    return { rbegin(ret), rend(ret) };
}
