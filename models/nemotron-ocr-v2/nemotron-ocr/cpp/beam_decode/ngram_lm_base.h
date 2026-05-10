// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <vector>

#include "language_model.h"

// #define USE_BOOST 1


typedef std::unordered_map<std::wstring, uint32_t> suffix_map_t;

/* Tells us the number of suffixes for a given ngram of order K
   Keys:
    1. NGram Order
    2. Prefix
    3. Suffix
Value:
    Count
*/
typedef std::unordered_map<std::wstring, suffix_map_t> string_suffix_map_t;
typedef std::vector<string_suffix_map_t> lookup_t;
/* Tells us the number of K-gram prefixes found for a given suffix
   Keys:
    1. Suffix
    2. NGram Order
    3. Prefix
Values:
    Count
*/
typedef std::vector<suffix_map_t> suffix_map_vec_t;
typedef std::unordered_map<std::wstring, suffix_map_vec_t> reverse_lookup_t;



extern const std::wstring WORD_END;
extern const std::wstring NUMERIC;
extern const std::wstring UNMODELED;

class NGramLMBase
    : public LanguageModel
{
public:
    virtual float_t ScoreTransition(const Prefix *p, token_t nextToken) const override;

protected:
    NGramLMBase(const std::string &dataFilePath, token_mapping_t tokenMapping);

    virtual float_t ScoreTransitionImpl(const std::wstring &prefix, const std::wstring &suffix) const = 0;

    bool ConvertToString(const Prefix *p, std::wstring &prefix) const;

    float_t GetPrefixSum(const std::wstring &prefix) const;

    lookup_t m_lookup;
    reverse_lookup_t m_reverseLookup;

    std::unordered_map<std::wstring, uint32_t> m_prefixSumLookup;
};

#if ! defined( USE_BOOST )
void save_ngram_data_file(const lookup_t& lookup, const reverse_lookup_t& reverseLookup, const std::string &output_path);
#else // USE_BOOST
void save_ngram_data_file(lookup_t lookup, reverse_lookup_t reverseLookup, const std::string &output_path);
#endif // USE_BOOST

inline float_t NGramLMBase::GetPrefixSum(const std::wstring &prefix) const
{
    auto iter = m_prefixSumLookup.find(prefix);

    if (iter == m_prefixSumLookup.end()) {
        return 0;
    } else {
        return iter->second;
    }
}
