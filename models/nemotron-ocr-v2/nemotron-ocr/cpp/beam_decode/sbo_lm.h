// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kn_lm.h"


class SBO_LanguageModel
    : public NGramLMBase
{
public:
    SBO_LanguageModel(const std::string &dataFilePath, token_mapping_t tokenMapping, float_t backoff);

protected:
    virtual float_t ScoreTransitionImpl(const std::wstring &prefix, const std::wstring &suffix) const override;

private:
    float_t m_backoff;
};
