// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ngram_lm_base.h"

#include <iostream>
#include <fstream>

#if defined( USE_BOOST )

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/unordered_map.hpp>

#endif // USE_BOOST

using namespace std;

const std::wstring WORD_END(1, 2);
const std::wstring NUMERIC(1, 3);
const std::wstring UNMODELED(1, 4);

struct LMStorage
{
    lookup_t Lookup;
    reverse_lookup_t ReverseLookup;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & Lookup;
        ar & ReverseLookup;
    }
};

void save_suffix_map(std::fstream& fs, const suffix_map_t& suffix_map)
{
    // write out number of elements for Lookup
    std::size_t suffix_map_count = suffix_map.size();
    fs.write((char*)(&suffix_map_count), sizeof(suffix_map_count));
    for (suffix_map_t::const_iterator reverse_lookup_it = suffix_map.begin(); reverse_lookup_it != suffix_map.end(); ++reverse_lookup_it)
    {
        // write out the key
        size_t key_len = reverse_lookup_it->first.length();
        fs.write((char*)(&key_len), sizeof(key_len));
        fs.write((char*)(reverse_lookup_it->first.data()), key_len * sizeof(wchar_t));

        // write out value
        fs.write((char*)(&reverse_lookup_it->second), sizeof(reverse_lookup_it->second));
    }
}

void save_lookup(std::fstream& fs, const lookup_t& lookup)
{
    // write out number of elements for Lookup
    std::size_t lookup_count = lookup.size();
    fs.write((char*)(&lookup_count), sizeof(lookup_count));
    for (lookup_t::const_iterator lookup_it = lookup.begin(); lookup_it != lookup.end(); ++lookup_it)
    {
        // write out element map size
        std::size_t map_elem_count = lookup_it->size();
        fs.write((char*)(&map_elem_count), sizeof(map_elem_count));

        for (string_suffix_map_t::const_iterator str_sfx_it = lookup_it->begin(); str_sfx_it != lookup_it->end(); ++str_sfx_it)
        {
            // write out key
            size_t key_len = str_sfx_it->first.length();
            fs.write((char*)(&key_len), sizeof(key_len));
            fs.write((char*)(str_sfx_it->first.data()), key_len * sizeof(wchar_t));
            save_suffix_map(fs, str_sfx_it->second);
        }
    }
}

void save_reverse_lookup(std::fstream& fs, const reverse_lookup_t& reverse_lookup)
{
    // write out number of elements for Lookup
    std::size_t reverse_lookup_count = reverse_lookup.size();
    fs.write((char*)(&reverse_lookup_count), sizeof(reverse_lookup_count));
    for (reverse_lookup_t::const_iterator reverse_lookup_it = reverse_lookup.begin(); reverse_lookup_it != reverse_lookup.end(); ++reverse_lookup_it)
    {
        // write out the key
        size_t key_len = reverse_lookup_it->first.length();
        fs.write((char*)(&key_len), sizeof(key_len));
        fs.write((char*)(reverse_lookup_it->first.data()), key_len * sizeof(wchar_t));

        // write out value vector length
        size_t val_vec_len = reverse_lookup_it->second.size();
        fs.write((char*)(&val_vec_len), sizeof(val_vec_len));

        for (suffix_map_vec_t::const_iterator val_vec_it = reverse_lookup_it->second.begin();
            val_vec_it != reverse_lookup_it->second.end();
            ++val_vec_it)
        {
            save_suffix_map(fs, *val_vec_it);
        }
    }
}

void load_suffix_map(std::fstream& fs, suffix_map_t& suffix_map)
{
    // read in number of elements
    std::size_t suffix_map_count = 0;
    fs.read((char*)(&suffix_map_count), sizeof(suffix_map_count));
    for (size_t suffix_map_index = 0; suffix_map_index < suffix_map_count; ++suffix_map_index )
    {
        // read in key
        std::size_t key_len = 0;
        fs.read((char*)(&key_len), sizeof(key_len));

        std::wstring wkey(key_len, 0);
        fs.read((char*)(wkey.data()), key_len * sizeof(wchar_t));
        uint32_t value = 0;
        fs.read((char*)(&value), sizeof(value));

        suffix_map.insert(std::make_pair(wkey, value));
    }
}

void load_lookup(std::fstream& fs, lookup_t& lookup)
{
    // read in number of elements
    std::size_t lookup_count = 0;
    fs.read((char*)(&lookup_count), sizeof(lookup_count));
    for (size_t lookup_index = 0; lookup_index < lookup_count; ++lookup_index)
    {
        std::size_t map_elem_count = 0;
        fs.read((char*)(&map_elem_count), sizeof(map_elem_count));

        lookup.push_back(string_suffix_map_t());
        string_suffix_map_t& str_sfx_map = lookup.back();

        for (size_t str_sfx_map_index = 0; str_sfx_map_index < map_elem_count; ++str_sfx_map_index)
        {
            std::size_t key_len = 0;
            fs.read((char*)(&key_len), sizeof(key_len));

            std::wstring wkey(key_len, 0);
            fs.read((char*)(wkey.data()), key_len * sizeof(wchar_t));
            str_sfx_map.insert(std::make_pair<wstring, suffix_map_t>(std::wstring(wkey), suffix_map_t()));
            suffix_map_t& suffix_map = str_sfx_map[wkey];

            load_suffix_map(fs, suffix_map);
        }
    }
}

void load_reverse_lookup(std::fstream& fs, reverse_lookup_t& reverse_lookup)
{
    // read in number of elements
    std::size_t reverse_lookup_count = 0;
    fs.read((char*)(&reverse_lookup_count), sizeof(reverse_lookup_count));
    for (size_t rev_lookup_index = 0; rev_lookup_index < reverse_lookup_count; ++rev_lookup_index )
    {
        // read in the key
        std::size_t key_len = 0;
        fs.read((char*)(&key_len), sizeof(key_len));

        std::wstring wkey(key_len, 0);
        fs.read((char*)(wkey.data()), key_len * sizeof(wchar_t));
        reverse_lookup.insert(std::make_pair(wkey, suffix_map_vec_t()));
        suffix_map_vec_t& val_vec = reverse_lookup[wkey];

        std::size_t val_vec_len = 0;
        fs.read((char*)(&val_vec_len), sizeof(val_vec_len));

        for (size_t val_vec_index = 0; val_vec_index < val_vec_len; ++val_vec_index)
        {
            val_vec.push_back(suffix_map_t());
            suffix_map_t& suffix_map = val_vec.back();
            load_suffix_map(fs, suffix_map);
        }
    }
}

#if ! defined( USE_BOOST )

NGramLMBase::NGramLMBase(const string &dataFilePath, token_mapping_t tokenMapping)
    : LanguageModel(move(tokenMapping))
{
    std::fstream in(dataFilePath, std::ios::in | std::ios::binary);
    load_lookup(in, m_lookup);
    load_reverse_lookup(in, m_reverseLookup);

    if (m_lookup.size() >= 10) {
        throw runtime_error("Only N-Grams of 9 or less are supported!");
    }

    for (auto &ngLevel : m_lookup) {
        for (auto &kvPrefixLevel : ngLevel) {
            uint32_t ct = 0;
            for (auto &kvSfx : kvPrefixLevel.second) {
                ct += kvSfx.second;
            }
            m_prefixSumLookup.emplace(kvPrefixLevel.first, ct);
        }
    }
}

void save_ngram_data_file(const lookup_t& lookup, const reverse_lookup_t& reverseLookup, const std::string &outputPath)
{
    std::fstream out(outputPath, std::ios::out | std::ios::binary);

    save_lookup(out, lookup);
    save_reverse_lookup(out, reverseLookup);
}

#else // USE_BOOST

NGramLMBase::NGramLMBase(const string &dataFilePath, token_mapping_t tokenMapping)
    : LanguageModel(move(tokenMapping))
{
    {
        ifstream dfStr(dataFilePath, ios_base::in | ios_base::binary);
        boost::archive::binary_iarchive ia(dfStr);

        LMStorage s;
        ia >> s;


        m_lookup = move(s.Lookup);

        m_reverseLookup = move(s.ReverseLookup);
    }

    if (m_lookup.size() >= 10) {
        throw runtime_error("Only N-Grams of 9 or less are supported!");
    }

    for (auto &ngLevel : m_lookup) {
        for (auto &kvPrefixLevel : ngLevel) {
            uint32_t ct = 0;
            for (auto &kvSfx : kvPrefixLevel.second) {
                ct += kvSfx.second;
            }
            m_prefixSumLookup.emplace(kvPrefixLevel.first, ct);
        }
    }
}

void save_ngram_data_file(lookup_t lookup, reverse_lookup_t reverseLookup, const std::string &outputPath)
{
    ofstream ofs(outputPath, ios_base::out | ios_base::binary);

    LMStorage s;
    s.Lookup = move(lookup);
    s.ReverseLookup = move(reverseLookup);

    boost::archive::binary_oarchive oa(ofs);
    oa << s;
}

#endif // USE_BOOST

float_t NGramLMBase::ScoreTransition(const Prefix *p, token_t nextToken) const
{
    std::wstring prefix;
    if (! ConvertToString(p, prefix)) {
        return NEG_INF;
    }

    const std::wstring *pSuffix = nullptr;

    if (nextToken != 1) {
        auto iter = m_tokenMapping.find(nextToken);
        if (iter == m_tokenMapping.end()) {
            pSuffix = &UNMODELED;
        } else {
            pSuffix = &iter->second;

            if (iswdigit(pSuffix->at(0))) {
                pSuffix = &NUMERIC;
            }
        }

    } else {
        pSuffix = &WORD_END;
    }

    float_t ret = ScoreTransitionImpl(prefix, *pSuffix);

    if (ret > 0) {
        return log(ret);
    } else {
        return NEG_INF;
    }
}

bool NGramLMBase::ConvertToString(const Prefix *p, std::wstring &prefix) const
{
    const Prefix *stk[10];
    int32_t sz = -1;
    const Prefix *curr = p;
    decltype(sz) mlSz{(int)m_lookup.size() - 2};
    while (curr && sz < mlSz) {
        stk[++sz] = curr;
        curr = curr->Parent;
    }

    // Either blank or empty prefix
    if (sz < 1) { return true; }

    --sz;
    for (; sz >= 0; --sz) {
        token_t tok = stk[sz]->Token;
        // End of word token, which maps to the null character
        if (tok == 1) {
            prefix.push_back(WORD_END[0]);
        } else if (tok == 0) {
            // Do nothing
        } else {
            auto iter = m_tokenMapping.find(tok);
            if (iter == m_tokenMapping.end()) {
                prefix += UNMODELED;
            } else {
                const std::wstring &wChar = iter->second;

                if (iswdigit(wChar[0])) {
                    prefix += NUMERIC;
                } else {
                    prefix += wChar;
                }
            }
        }
    }

    return true;
}
