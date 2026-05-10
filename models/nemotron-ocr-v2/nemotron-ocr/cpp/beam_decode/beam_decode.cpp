// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "beam_decode.h"

#include <vector>
#include <deque>
#include <limits>
#include <memory>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <chrono>

#include "../common.h"
#include "prefix.h"
#include "log_sum_exp.h"
#include "sbo_lm.h"

using namespace std;

template<typename scalar_t>
using pred_seq_t = torch::TensorAccessor<scalar_t, 2>;

struct PrefixScore
{
    float_t lProbBlank;
    float_t lProbChar;
    // float_t raw_lProbBlank;
    // float_t raw_lProbChar;
    mutable float_t _lProb;

    PrefixScore(float_t lProbBlank = NEG_INF /* log P(0) */, float_t lProbChar = NEG_INF /* log P(0) */)
        : lProbBlank(lProbBlank), lProbChar(lProbChar), _lProb(NEG_INF)
        //   , raw_lProbBlank(lProbBlank), raw_lProbChar(lProbChar)
    {}

    float_t get_lScore() const {
        if (_lProb == NEG_INF) {
            _lProb = log_sum_exp(lProbBlank, lProbChar);
        }
        return _lProb;
    }

    // float_t get_raw_lScore() const {
    //     return log_sum_exp(raw_lProbBlank, raw_lProbChar);
    // }
};

typedef std::unordered_map<Prefix*, PrefixScore> PrefixMap;
typedef std::pair<Prefix*, PrefixScore> BeamItem;
typedef std::vector<BeamItem> Beam;

/*
    Allows us to get an estimate of the vision model confidence, irrespective of how the language
    model guided the decoding. NOTE: This scoring could follow an entirely different path than
    the returned decoded sequence.
*/
template<typename scalar_t>
scalar_t get_vision_confidence(const pred_seq_t<scalar_t> &logProbs, scalar_t minProb)
{
    const int64_t T = logProbs.size(0);
    const int64_t S = logProbs.size(1);

    scalar_t ret = 0; // log(1)

    for (size_t t = 0; t < T; ++t) {
        float_t maxP = logProbs[t][0];
        int64_t maxC = 0;
        for (int64_t c = 1; c < S; ++c) {
            float_t p = logProbs[t][c];
            if (p > maxP) {
                maxP = p;
                maxC = c;
            }
        }
        ret += maxP;
        // Ignore everything past the sequence terminator
        if (maxC == 1) {
            break;
        }

        if (ret < minProb) {
            break;
        }
    }

    return ret;
}


template<typename scalar_t>
pair<vector<token_t>, float_t>
    ctc_beam_decode_impl(const pred_seq_t<scalar_t> &probs, const int64_t beamSize,
                         const int64_t blank, scalar_t minProb,
                         const LanguageModel &langModel, scalar_t lmWeight)
{
    if (blank != 0) {
        throw runtime_error("Currently, only ordinal 0 supported for the blank prediction");
    }

    const int64_t T = probs.size(0);
    const int64_t S = probs.size(1);

    // NOTE: In log space, the following is true:
    // 1. Adding two probabilities: log_sum_exp(l_p_a, l_p_b)
    // 2. Multiplying two probabilities: l_p_a + l_p_b
    // 3. log P(0) = -inf
    // 4. log P(1) = 0

    // Convert to log-space
    if (minProb > 0) {
        minProb = log(minProb);
    } else {
        minProb = NEG_INF;
    }

    auto retScore = get_vision_confidence(probs, minProb);

    if (retScore < minProb) {
        return { {}, NEG_INF };
    }

    PrefixAllocator prefixAlloc;

    Beam beam;
    beam.emplace_back(prefixAlloc.GetPrefix(), PrefixScore{0, NEG_INF}); // Add a dummy first node

    Beam terminated;

    typedef tuple<Prefix*, token_t> lm_cache_key_t;
    unordered_map<lm_cache_key_t, float_t> lmScoreCache;

    for (int64_t t = 0; t < T; ++t) {
        PrefixMap nextBeam;

        // Add all of the completed paths to the next beam.
        // This allows us to accumulate new paths into these,
        // but otherwise not process them
        for (const BeamItem &prevNode : beam) {
            if (prevNode.first->Token == 1) {
                nextBeam.insert(prevNode);
            }
        }

        // Loop over vocab
        for (int64_t s = 0; s < S; ++s) {
            float_t lpEmit = probs[t][s];

            if (lpEmit < minProb) {
                continue;
            }

            for (const BeamItem &prevNode : beam) {
                Prefix *prevPrefix = prevNode.first;
                const PrefixScore &prevScore = prevNode.second;

                // Ignore already completed paths
                if (prevPrefix->Token == 1) {
                    continue;
                }

                // Ignore impossible paths
                if (prevScore.lProbBlank == NEG_INF && prevScore.lProbChar == NEG_INF) {
                    continue;
                }

                // If we propose a blank the prefix doesn't change.
                // Only the probability of ending in blank gets updated.
                if (s == blank) {
                    PrefixScore &score = nextBeam[prevPrefix];
                    score.lProbBlank     = log_sum_exp(score.lProbBlank    , prevScore.lProbBlank     + lpEmit, prevScore.lProbChar     + lpEmit);
                    // score.raw_lProbBlank = log_sum_exp(score.raw_lProbBlank, prevScore.raw_lProbBlank + lpEmit, prevScore.raw_lProbChar + lpEmit);
                    continue;
                }

                // Extend the prefix by the new character s and add it to the beam.
                // Only the probability of not ending in blank gets updated.
                token_t prevToken = prevPrefix->Token;

                // NOTE: We always create a new prefix regardless of duplication because the PrefixScore
                // is simultaneously tracking prefixes that do and don't end in a blank. And it's those
                // that end in a blank that would cause the prefix to be extended.
                auto extendPrefix = prefixAlloc.GetPrefix(s, prevPrefix);

                // Evaluate the language model, but use the cache if we've already considered this string before
                auto lmCacheItem = make_tuple(prevPrefix, s);
                auto lmCacheIter = lmScoreCache.find(lmCacheItem);
                float_t lpLang = 0;
                if (lmCacheIter == lmScoreCache.end()) {
                    lpLang = langModel.ScoreTransition(prevPrefix, s);
                    lpLang *= lmWeight;
                    lmCacheIter = lmScoreCache.emplace(lmCacheItem, lpLang).first;
                }
                lpLang = lmCacheIter->second;

                PrefixScore &extendScore = nextBeam[extendPrefix];
                // Remember, adding two log probabilities is equivalent to multiplying two probabilities
                if (s != prevToken) {
                    extendScore.lProbChar     = log_sum_exp(extendScore.lProbChar,     prevScore.lProbBlank     + lpEmit + lpLang, prevScore.lProbChar     + lpEmit + lpLang);
                    // extendScore.raw_lProbChar = log_sum_exp(extendScore.raw_lProbChar, prevScore.raw_lProbBlank + lpEmit         , prevScore.raw_lProbChar + lpEmit         );
                } else {
                    // We don't include the previous probability of not ending in blank if s is repeated at the end. The CTC
                    // algorithm merges characters not separated by a blank.
                    extendScore.lProbChar     = log_sum_exp(extendScore.lProbChar    , prevScore.lProbBlank     + lpEmit + lpLang);
                    // extendScore.raw_lProbChar = log_sum_exp(extendScore.raw_lProbChar, prevScore.raw_lProbBlank + lpEmit         );
                }

                // If the token is repeated, we also have to deal with the unchanged prefix since repeated characters are collapsed
                if (s == prevToken) {
                    PrefixScore &collapseScore = nextBeam[prevPrefix];
                    collapseScore.lProbChar     = log_sum_exp(collapseScore.lProbChar    , prevScore.lProbChar     + lpEmit);
                    // collapseScore.raw_lProbChar = log_sum_exp(collapseScore.raw_lProbChar, prevScore.raw_lProbChar + lpEmit);
                }

            }
        }

        Beam vecNextBeam(begin(nextBeam), end(nextBeam));

        if (vecNextBeam.size() > beamSize) {
            partial_sort(begin(vecNextBeam), begin(vecNextBeam) + beamSize, end(vecNextBeam),
                [] (const BeamItem &a, const BeamItem &b) {
                    return a.second.get_lScore() > b.second.get_lScore();
                }
            );
            vecNextBeam.resize(beamSize);
        }

        beam = move(vecNextBeam);
    }

    // Find the best raw score
    const BeamItem *bestItem = nullptr;
    // for (const BeamItem &b : beam) {
    //     if (bestItem == nullptr or b.second.get_raw_lScore() > bestItem->second.get_raw_lScore()) {
    //         bestItem = &b;
    //     }
    // }
    if (! beam.empty()) {
        bestItem = &beam[0];
    }

    if (bestItem != nullptr) {
        auto retList = bestItem->first->ToList();

        return { move(retList), retScore };
    } else {
        return { {}, NEG_INF };
    }
}

typedef std::pair<Prefix*, float_t> RegBeamItem;

bool operator<(const RegBeamItem &a, const RegBeamItem &b) {
    return a.second > b.second;
}

template<typename scalar_t>
pair<vector<token_t>, float_t>
    reg_beam_decode_impl(const pred_seq_t<scalar_t> &logProbs, const int64_t beamSize,
                         scalar_t minProb,
                         const LanguageModel &langModel, scalar_t lmWeight)
{
    const int64_t T = logProbs.size(0);
    const int64_t S = logProbs.size(1);

    // NOTE: In log space, the following is true:
    // 1. Adding two probabilities: log_sum_exp(l_p_a, l_p_b)
    // 2. Multiplying two probabilities: l_p_a + l_p_b
    // 3. log P(0) = -inf
    // 4. log P(1) = 0

    // Convert to log-space
    if (minProb > 0) {
        minProb = log(minProb);
    } else {
        minProb = NEG_INF;
    }

    auto retScore = get_vision_confidence(logProbs, minProb);

    if (retScore < minProb) {
        return { {}, NEG_INF };
    }

    PrefixAllocator prefixAlloc;

    vector<RegBeamItem> beam, nextBeam;
    beam.emplace_back(prefixAlloc.GetPrefix(), 0); // log(1) = 0

    for (int64_t t = 0; t < T && !beam.empty(); ++t) {
        nextBeam.clear();

        auto addToBeam = [&nextBeam, beamSize] (const RegBeamItem &rbi) {
            nextBeam.push_back(rbi);
        };

        // Expand each path in the beam
        for (const RegBeamItem &prevNode : beam) {
            if (prevNode.first->Token == 1) {
                // Move completed paths along without processing further
                addToBeam(prevNode);
                continue;
            }

            Prefix *prevPrefix = prevNode.first;
            float_t prevScore = prevNode.second;

            // Loop over vocab
            for (int64_t s = 0; s < S; ++s) {
                float_t lpEmit = logProbs[t][s];

                if (lpEmit < minProb) {
                    // The probability dropped below threshold, so stop processing this path
                    continue;
                }

                auto extendPrefix = prefixAlloc.GetPrefix(s, prevPrefix);

                float_t lpLang = langModel.ScoreTransition(prevPrefix, s);

                float_t lpNext = prevScore + lpLang + lpEmit;

                addToBeam({extendPrefix, lpNext});
            }
        }

        if (nextBeam.size() > beamSize) {
            // Find the top-k items, and then truncate the rest
            partial_sort(begin(nextBeam), begin(nextBeam) + beamSize, end(nextBeam));
            nextBeam.resize(beamSize);
        }

        std::swap(beam, nextBeam);
    }

    if (!beam.empty()) {
        // The highest probability element will always be in the back
        RegBeamItem rbi{ nullptr, NEG_INF };
        for (auto &rb : beam) {
            if (rbi.first == nullptr || rb.second > rbi.second) {
                rbi = rb;
            }
        }

        auto retList = rbi.first->ToList();

        return { move(retList), retScore };
    } else {
        return { {}, NEG_INF };
    }
}



template<typename scalar_t>
void dp_beam_decode_impl(const torch::TensorAccessor<scalar_t, 3> &probsAccess,
                         torch::TensorAccessor<int64_t, 2> retAccess,
                         torch::TensorAccessor<scalar_t, 1> confAccess,
                         int64_t beamSize, int64_t blank,
                         scalar_t minProb,
                         const LanguageModel *langModel,
                         scalar_t lmWeight,
                         bool combineDuplicates)
{
    const int64_t N = probsAccess.size(0);

    #pragma omp parallel for num_threads(8)
    for (int64_t i = 0; i < N; ++i) {
        vector<token_t> seq;
        float_t lConf;
        if (combineDuplicates) {
            tie(seq, lConf) = ctc_beam_decode_impl(probsAccess[i], beamSize, blank,
                                                   minProb,
                                                   *langModel, lmWeight);
        } else {
            tie(seq, lConf) = reg_beam_decode_impl(probsAccess[i], beamSize,
                                                   minProb,
                                                   *langModel, lmWeight);
        }

        int64_t sz = min<int64_t>(seq.size(), retAccess.size(1));

        for (int64_t k = 0; k < sz; ++k) {
            retAccess[i][k] = seq[k];
        }

        confAccess[i] = exp(lConf);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
    beam_decode(torch::Tensor probs, int64_t beamSize, int64_t blank,
                float minProb,
                const LanguageModel *langModel,
                float lmWeight,
                bool combineDuplicates)
{
    if (langModel == nullptr) {
        langModel = &NullLanguageModel;
    }

    auto tStart = chrono::high_resolution_clock::now();

    probs = probs.contiguous();

    bool collapse = false;
    if (probs.dim() == 2) {
        // N,T,C
        probs = probs.unsqueeze(0);
        collapse = true;
    }

    probs = probs.log();

    torch::Tensor ret = torch::ones({ probs.size(0), probs.size(1) }, torch::kInt64);
    torch::Tensor conf = torch::zeros({ probs.size(0) }, probs.options());

    auto retAccess = ret.accessor<int64_t, 2>();

    AT_DISPATCH_FLOATING_TYPES(
        probs.scalar_type(),
        "cpu_beam_decode",
        ([&] {
            dp_beam_decode_impl(
                probs.accessor<scalar_t, 3>(),
                retAccess,
                conf.accessor<scalar_t, 1>(),
                beamSize, blank,
                static_cast<scalar_t>(minProb),
                langModel,
                static_cast<scalar_t>(lmWeight),
                combineDuplicates
            );
        })
    );

    if (collapse) {
        ret = ret.squeeze(0);
        conf = conf[0];
    }

    auto tEnd = chrono::high_resolution_clock::now();

    typedef chrono::duration<double, std::milli> tp_t;
    tp_t totalElapsed = tEnd - tStart;

    cout << "Beam Decode " << probs.size(0) << " - "
         << "Total: " << totalElapsed.count() << "ms"
         << endl;

    return { ret, conf };
}

std::unique_ptr<LanguageModel> create_sbo_lm(const std::string &dataFilePath, token_mapping_t tokenMapping, float_t backoffWeight)
{
    return make_unique<SBO_LanguageModel>(dataFilePath, move(tokenMapping), backoffWeight);
}
