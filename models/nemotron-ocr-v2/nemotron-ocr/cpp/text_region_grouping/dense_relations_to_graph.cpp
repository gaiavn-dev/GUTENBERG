// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "text_region_grouping.h"

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <stack>
#include <numeric>
#include <vector>

using namespace std;

PhraseList rel_list_to_phrases(const relations_list_t &relList)
{
    PhraseList ret;
    ret.reserve(relList.size());

    for (const text_line_t &line : relList) {
        TextLine tl;
        tl.reserve(line.size());

        for (const auto &rel : line) {
            tl.push_back(get<0>(rel));
        }

        ret.push_back({ move(tl) });
    }

    return ret;
}

template<typename rel_to_2_from_map_t, typename T>
relations_list_t rel_chain_to_groups(const rel_to_2_from_map_t &inChain, int64_t numRegions, const T *inProbs);

template<typename T>
relations_list_t dense_relations_to_graph_impl(torch::Tensor relationsTensor)
{
    if (relationsTensor.size(0) == 0) {
        return relations_list_t{};
    }

    if (relationsTensor.size(0) != relationsTensor.size(1)) {
        throw std::runtime_error("The relations tensor must be a square matrix!");
    }

    // Each row `i` of `relationsTensor` is a probability distribution of going from word `i` to word `k`
    // If we find the maximum confidence into each word `k`, it tells us the strongest connection
    // from `i` to `k`.
    // So, `maxRelTensor` tells us the connection strength of the strongest connection coming into word `k`,
    // and `fromIdxTensor` tells us the index of word `i` that has this connection
    auto relations = relationsTensor.accessor<T, 2>();

    const int64_t numRegions = relationsTensor.size(0);
    torch::Tensor fromIdxsTensor = torch::full({ numRegions }, -1, torch::kInt64);
    torch::Tensor fromProbsTensor = torch::zeros({ numRegions }, relationsTensor.options());

    // Use `data_ptr` here because these tensors are 1-dimensional contiguous arrays, which saves us
    // a multiply+add for each access
    auto fromIdxs = fromIdxsTensor.data_ptr<int64_t>();
    auto fromProbs = fromProbsTensor.data_ptr<T>();

    for (int64_t fromIdx = 0; fromIdx < numRegions; ++fromIdx) {
        auto fromRel = relations[fromIdx];

        for (int64_t toIdx = 0; toIdx < numRegions; ++toIdx) {
            auto relProb = fromRel[toIdx];

            if (relProb >= 0.5) {
                T &maxProb = fromProbs[toIdx];
                if (fromIdxs[toIdx] == -1 || relProb > maxProb) {
                    fromIdxs[toIdx] = fromIdx;
                    maxProb = relProb;
                }
                // Because each row sums to 1, it's only possible for <= 1 columns to have
                // a value above 0.5
                break;
            }
        }
    }

    return rel_chain_to_groups(fromIdxs, numRegions, fromProbs);
}

relations_list_t dense_relations_to_graph_with_probs(torch::Tensor relationsTensor)
{
    relations_list_t ret;
    AT_DISPATCH_FLOATING_TYPES(
        relationsTensor.scalar_type(),
        "dense_relations_to_graph",
        ([&] {
            ret = dense_relations_to_graph_impl<scalar_t>(relationsTensor);
        })
    );
    return ret;
}

PhraseList dense_relations_to_graph(torch::Tensor relations)
{
    return rel_list_to_phrases(dense_relations_to_graph_with_probs(relations));
}

template<typename T>
relations_list_t sparse_relations_to_graph_impl(torch::Tensor relationsTensor, torch::Tensor neighborIdxsTensor)
{
    if (relationsTensor.size(0) == 0) {
        return relations_list_t{};
    }

    auto maxRelsTensor = torch::zeros({ relationsTensor.size(0) }, relationsTensor.options());
    auto fromIdxsTensor = torch::full({ relationsTensor.size(0) }, -1, torch::kInt64);

    auto relations = relationsTensor.accessor<T, 2>();
    auto neighborIdxs = neighborIdxsTensor.accessor<int64_t, 2>();
    auto maxRels = maxRelsTensor.data_ptr<T>();
    auto fromIdxs = fromIdxsTensor.data_ptr<int64_t>();

    const int64_t N = relationsTensor.size(0);
    const int64_t K = relationsTensor.size(1);

    // Refer to `dense_relations_to_graph` for the reasoning behind this. The only difference here
    // is the indirection due to sparsity. At the completion of this double loop,
    // `maxRelsTensor` and `fromIdxTensor` are of identical form to the dense case.
    for (int64_t fromIdx = 0; fromIdx < N; ++fromIdx) {
        auto fromNeighborIdxs = neighborIdxs[fromIdx].data();
        auto fromRelations = relations[fromIdx].data();

        // Skip the null column
        for (int64_t c = 1; c < K; ++c) {
            // All of these values will be offset by +1 to account for the null column
            int64_t toIdx = fromNeighborIdxs[c] - 1;
            // The relations tensor already has the null column stripped off
            T toProb = fromRelations[c];

            if (toProb > 0.5f) {
                T &bestProb = maxRels[toIdx];
                if (toProb > bestProb) {
                    bestProb = toProb;
                    fromIdxs[toIdx] = fromIdx;
                }
                // Due to the softmax, only one value could ever be >0.5, if any,
                // so if we've encountered this value, then we're done with this `fromIdx`
                break;
            }
        }
    }

    return rel_chain_to_groups(fromIdxs, N, maxRels);
}

relations_list_t sparse_relations_to_graph(torch::Tensor relationsTensor, torch::Tensor neighborIdxs)
{
    relations_list_t ret;

    AT_DISPATCH_FLOATING_TYPES(
        relationsTensor.scalar_type(),
        "sparse_relations_to_graph",
        ([&] {
            ret = sparse_relations_to_graph_impl<scalar_t>(relationsTensor, neighborIdxs);
        })
    );

    return ret;
}

template<typename rel_to_2_from_map_t, typename T>
relations_list_t rel_chain_to_groups(const rel_to_2_from_map_t &inChain, const int64_t numRegions, const T *inProbs)
{
    // inChain is a vector over the relations that tells us, for a given position `i`,
    // the strongest relation `k` leading into that, if any, otherwise -1.
    // So if `inChain[5] == 2`, this means that region `k==2` connects to region `i==5`.
    // It's also mandatory that the elements in inChain != -1 form a bijection
    // between from/to (e.g. the same from index can't be used twice)

    // Create a mapping that goes from word `fromIdx` to word `toIdx`, which is the
    // reverse mapping of inChain
    auto outChainTensor = torch::full({ numRegions }, -1, torch::kInt64);
    auto outChain = outChainTensor.data_ptr<int64_t>();

    auto outProbsTensor = torch::ones({ numRegions }, torch::kFloat);
    auto outProbs = outProbsTensor.data_ptr<float>();

    for (int64_t toIdx = 0; toIdx < numRegions; ++toIdx) {
        int64_t fromIdx = inChain[toIdx];
        if (fromIdx != -1) {
            outChain[fromIdx] = toIdx;
            outProbs[fromIdx] = static_cast<float>(inProbs[toIdx]);
        }
    }

    std::vector<bool> processed; processed.resize(numRegions, false);

    text_line_t currChain; currChain.reserve(32);
    relations_list_t groups;

    for (int64_t toIdx = 0; toIdx < numRegions; ++toIdx) {
        int64_t fromIdx = inChain[toIdx];

        if (fromIdx == -1 || processed[toIdx]) {
            continue;
        }

        processed[toIdx] = true;
        currChain.clear();
        currChain.emplace_back(toIdx, outProbs[fromIdx]);

        int64_t currIdx = toIdx;
        while (true) {
            fromIdx = inChain[currIdx];
            // The second check ensures that we don't encounter any cycles
            if (fromIdx == -1 || processed[fromIdx]) {
                break;
            }

            processed[fromIdx] = true;
            currChain.emplace_back(fromIdx, outProbs[fromIdx]);
            currIdx = fromIdx;
        }

        // At this point, `currChain` contains all of the indices from `toIdx` (index 0) backward.
        // So, we can initialize the group with the reverse iterator to the current chain
        text_line_t group{ std::rbegin(currChain), std::rend(currChain) };

        // However, we also need to harvest all of the indices from `toIdx` forward
        int64_t nextIdx = toIdx;
        while (true) {
            int64_t nextToIdx = outChain[nextIdx];
            // Same as before, second check will break cycles
            if (nextToIdx == -1 || processed[nextToIdx]) {
                break;
            }

            processed[nextToIdx] = true;
            group.emplace_back(nextToIdx, static_cast<float>(inProbs[nextToIdx]));
            nextIdx = nextToIdx;
        }

        groups.push_back(move(group));
    }

    // Now add in the stragglers
    for (int64_t wIdx = 0; wIdx < numRegions; ++wIdx) {
        if (! processed[wIdx]) {
            groups.push_back({ { wIdx, 1.0f } });
        }
    }

    return groups;
}
