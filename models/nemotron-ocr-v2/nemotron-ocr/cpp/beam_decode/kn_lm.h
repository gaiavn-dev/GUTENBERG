// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <vector>

#include "ngram_lm_base.h"


class KN_LanguageModel
    : public NGramLMBase
{
public:
    KN_LanguageModel(const std::string &dataFilePath, token_mapping_t tokenMapping, float_t knDelta);

protected:
    virtual float_t ScoreTransitionImpl(const std::wstring &prefix, const std::wstring &suffix) const override;

private:
    float_t ScoreUnigram(const std::wstring &uni) const;
    float_t ScoreTransition(const std::wstring &prefix, const std::wstring &suffix) const;

    float_t m_knDelta;
};
