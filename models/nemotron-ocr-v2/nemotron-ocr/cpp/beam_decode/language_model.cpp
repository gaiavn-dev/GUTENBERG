// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "language_model.h"

#include <locale>
#include <codecvt>

using namespace std;

const NullLanguageModel_t NullLanguageModel;

NullLanguageModel_t::NullLanguageModel_t()
    : LanguageModel({})
{
}

TokenMappingWrapper::TokenMappingWrapper(token_mapping_t mapping)
    : token_mapping(move(mapping))
{
    for (const auto &mp : token_mapping) {
        if (mp.second.size() == 1) {
            wchar_t c = mp.second.front();
            reverse_token_mapping.emplace(c, mp.first);
        }
    }
}

TokenMappingWrapper::Ptr create_token_mapping(token_mapping_t tokenMapping)
{
    return make_shared<TokenMappingWrapper>(move(tokenMapping));
}


template<typename token_t>
vector<tuple<wstring, float>>
    decode_sequences_impl(torch::Tensor tokens, const TokenMappingWrapper *tokenMapping,
                          c10::optional<torch::Tensor> probs)
{
    const token_mapping_t &mapping = tokenMapping->token_mapping;

    auto tokensAccess = tokens.accessor<token_t, 2>();

    torch::Tensor pTens = probs.value_or(torch::ones({ tokens.size(0) }, torch::kFloat32));
    if (pTens.dim() == 1) {
        pTens = pTens.unsqueeze(1);
    }

    auto probsAccess = pTens.accessor<float, 2>();

    const int64_t B = tokens.size(0);
    const int64_t T = tokens.size(1);

    vector<tuple<wstring, float>> ret;

    for (int64_t b = 0; b < B; ++b) {
        wstring buff;

        float logProb = 0.0f; // log 1
        bool done = false;
        for (int64_t t = 0; t < T && ! done; ++t) {
            typename token_mapping_t::key_type tokIdx = tokensAccess[b][t];

            if (t < probsAccess.size(1)) {
                logProb += log(probsAccess[b][t]);
            }

            switch (tokIdx) {
                case 0:
                    // Blank char
                    continue;
                case 1:
                    // End of sequence char
                    done = true;
                    break;
                case 2:
                    buff.push_back('^');
                    break;
                default:
                    auto iter = mapping.find(tokIdx);
                    if (iter == mapping.end()) {
                        throw std::runtime_error("The token mapping doesn't contain an entry for index " + to_string(tokIdx));
                    }
                    buff += iter->second;
                    break;
            }
        }

        ret.emplace_back(move(buff), exp(logProb));
    }

    return ret;
}

vector<tuple<wstring, float>>
    decode_sequences(torch::Tensor tokens, const TokenMappingWrapper *tokenMapping,
                     c10::optional<torch::Tensor> probs)
{
    if (tokens.dim() != 2) {
        throw std::runtime_error("`tokens` must be 2-dimensions of type B,T!");
    }

    if (tokenMapping == nullptr) {
        throw std::runtime_error("Cannot supply a null token mapping!");
    }

    const token_mapping_t &mapping = tokenMapping->token_mapping;

    if (mapping.empty()) {
        throw std::runtime_error("The token mapping hasn't been initialized!");
    }

    if (probs.has_value()) {
        if (probs.value().scalar_type() != torch::kFloat32) {
            throw std::runtime_error("If the probability distribution is specified, then it must be of type `torch.float32`");
        }
        if (probs.value().size(0) != tokens.size(0)) {
            throw std::runtime_error("The probability distribution batch size doesn't match the tokens batch size!");
        }
        if (probs.value().dim() == 2 && probs.value().size(1) != tokens.size(1)) {
            throw std::runtime_error("Invalid probability distribution shape!");
        }
    }

    vector<tuple<wstring, float>> ret;

    AT_DISPATCH_INTEGRAL_TYPES(
        tokens.scalar_type(),
        "decode_sequences_impl",
        ([&] {
            ret = decode_sequences_impl<scalar_t>(tokens, tokenMapping, probs);
        })
    );

    return ret;
}


std::string ws2s(const std::wstring& wstr)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;

    return converterX.to_bytes(wstr);
}

