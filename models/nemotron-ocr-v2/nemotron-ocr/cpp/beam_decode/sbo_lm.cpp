// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sbo_lm.h"

#include <assert.h>

// Reference paper: https://www.aclweb.org/anthology/D07-1090.pdf


SBO_LanguageModel::SBO_LanguageModel(const std::string &dataFilePath, token_mapping_t tokenMapping, float_t backoff)
    : NGramLMBase(dataFilePath, move(tokenMapping)), m_backoff(backoff)
{
}

float SBO_LanguageModel::ScoreTransitionImpl(const std::wstring &prefix, const std::wstring &suffix) const
{
    auto lIter = m_lookup[prefix.size() + 1].find(prefix);

    // This prefix doesn't exist. Shrink it!
    if (lIter == m_lookup[prefix.size() + 1].end()) {
        return m_backoff * ScoreTransitionImpl({ begin(prefix) + 1, end(prefix) }, suffix);
    }

    const suffix_map_t &suffixMap = lIter->second;

    auto sfIter = suffixMap.find(suffix);

    if (sfIter == suffixMap.end()) {
        // This is a novel character entirely!
        if (prefix.empty()) {
            return 1e-8;
        } else {
            return m_backoff * ScoreTransitionImpl({ begin(prefix) + 1, end(prefix) }, suffix);
        }
    }

    float_t ctSuffix = sfIter->second;
    float_t ctNgram = GetPrefixSum(prefix);

    float_t score = ctSuffix / ctNgram;

    assert(score >= 0 && score <= 1);

    return score;
}
