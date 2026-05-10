// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "kn_lm.h"

using namespace std;


KN_LanguageModel::KN_LanguageModel(const string &dataFilePath, token_mapping_t tokenMapping, float_t knDelta)
    : NGramLMBase(dataFilePath, move(tokenMapping)), m_knDelta(knDelta)
{
}

float KN_LanguageModel::ScoreTransitionImpl(const std::wstring &prefix, const std::wstring &suffix) const
{
    if (prefix.empty()) {
        return ScoreUnigram(suffix);
    } else {
        return ScoreTransition(prefix, suffix);
    }
}

float_t KN_LanguageModel::ScoreUnigram(const std::wstring &uni) const
{
    auto lIter = m_lookup[1].find(L""s);
    if (lIter == m_lookup[1].end()) {
        throw std::runtime_error("Unigrams not supported by this model!");
    }

    auto uniIter = lIter->second.find(uni);
    float_t ctUni = 1e-8;
    if (uniIter != lIter->second.end()) {
        ctUni = uniIter->second;
    }

    float_t ctSuffixes = GetPrefixSum(L""s);

    return ctUni / ctSuffixes;
}

float_t KN_LanguageModel::ScoreTransition(const std::wstring &prefix, const std::wstring &suffix) const
{
    if (prefix.empty()) {
        // The number of distinct bigrams that end with this token
        auto rlIter = m_reverseLookup.find(suffix);

        float_t ctEndingBigrams = 0;
        if (rlIter != m_reverseLookup.end()) {
            ctEndingBigrams = rlIter->second[2].size();
        }

        float_t ctAllBigrams = m_lookup[2].size();

        return ctEndingBigrams / ctAllBigrams;
    }

    auto lIter = m_lookup[prefix.size() + 1].find(prefix);
    float_t ctUqSuffixes = 0;
    float_t ctSuffixes = 0;
    float_t ctSuffix = 0;
    if (lIter != m_lookup[prefix.size() + 1].end()) {
        ctUqSuffixes = lIter->second.size();

        ctSuffixes = GetPrefixSum(prefix);

        auto sIter = lIter->second.find(suffix);
        if (sIter != lIter->second.end()) {
            ctSuffix = sIter->second;
        }
    }

    float_t factor = 0;
    float_t main = 0;
    if (ctSuffixes != 0) {
        factor = m_knDelta * ctUqSuffixes / ctSuffixes;
        // TODO: Figure out how to make this call without copying the string!
        factor *= ScoreTransition({begin(prefix) + 1, end(prefix)}, suffix);

        main = max<float_t>(ctSuffix - m_knDelta, 0) / ctSuffixes;
    }

    float_t total = main + factor;

    return total;
}
