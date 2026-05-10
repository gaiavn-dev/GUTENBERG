// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <torch/torch.h>

#include "prefix.h"
#include "log_sum_exp.h"

typedef std::unordered_map<int64_t, std::wstring> token_mapping_t;
typedef std::unordered_map<wchar_t, int64_t> reverse_token_mapping_t;


class LanguageModel
{
public:
    virtual ~LanguageModel() {}

    virtual float_t ScoreTransition(const Prefix *p, token_t nextToken) const = 0;

    const token_mapping_t &TokenMapping() const { return m_tokenMapping; }

protected:
    LanguageModel(token_mapping_t tokenMapping)
        : m_tokenMapping(std::move(tokenMapping))
    {}

    token_mapping_t m_tokenMapping;
};


class NullLanguageModel_t
    : public LanguageModel
{
public:
    NullLanguageModel_t();

    virtual float_t ScoreTransition(const Prefix *p, token_t nextToken) const override
    {
        // log P(1)
        // Which means the probability is unchanged
        return 0;
    }
};

extern const NullLanguageModel_t NullLanguageModel;

struct TokenMappingWrapper
{
    typedef std::shared_ptr<TokenMappingWrapper> Ptr;

    TokenMappingWrapper(token_mapping_t mapping);

    token_mapping_t token_mapping;
    reverse_token_mapping_t reverse_token_mapping;
};

TokenMappingWrapper::Ptr create_token_mapping(token_mapping_t tokenMapping);

std::vector<std::tuple<std::wstring, float>>
    decode_sequences(torch::Tensor tokens, const TokenMappingWrapper *tokenMapping,
                     c10::optional<torch::Tensor> probs = torch::nullopt);
