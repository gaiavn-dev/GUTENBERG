// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <memory>
#include <vector>
#include <unordered_map>
#include <list>

typedef int32_t token_t;

class Prefix;

// typedef std::shared_ptr<Prefix> PrefixPtr;

class Prefix
{
public:
    token_t Token;
    Prefix *Parent;

    Prefix(token_t token = 0 /* blank */, Prefix *parent = nullptr)
        : Token(token), Parent(parent)
    {}

    std::vector<token_t> ToList() const;

    size_t size() const;
};


///// Borrowed from Boost libraries
template<typename T>
void hash_combine(size_t & seed, T const& v)
{
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
/////

namespace std {
template<>
struct hash<Prefix*>
{
    size_t operator()(const Prefix *p) const noexcept
    {
        size_t seed = 0;

        while (p) {
            if (p->Token != 0) {
                hash_combine(seed, p->Token);
            }
            p = p->Parent;
        }
        return seed;
    }
};

template<>
struct hash<tuple<Prefix*, token_t>>
{
    size_t operator()(const tuple<Prefix*, token_t> &t) const noexcept
    {
        size_t seed = 0;
        hash_combine(seed, get<0>(t));
        hash_combine(seed, get<1>(t));
        return seed;
    }
};

template<>
struct equal_to<Prefix*>
{
    bool operator()(const Prefix *a, const Prefix *b) const noexcept
    {
        while (a != nullptr && b != nullptr) {
            if (a->Token != b->Token) {
                return false;
            }
            a = a->Parent;
            b = b->Parent;
        }
        // If one chain is shorter than the other
        return a == b;
    }
};
}

inline size_t Prefix::size() const
{
    size_t ret = 0;
    auto p = this;
    while (p != nullptr) {
        ret += 1;
        p = p->Parent;
    }
    return ret;
}


class PrefixAllocator
{
public:
    PrefixAllocator() = default;
    ~PrefixAllocator();

    template<typename ...Args>
    Prefix *GetPrefix(Args&& ...ctorArgs);

private:
    void AllocateNextBuffer();

    std::list<Prefix*> m_buffers;
    size_t m_allocSize = 0;
    size_t m_currOff = 0;
};

inline PrefixAllocator::~PrefixAllocator()
{
    for (auto p : m_buffers) {
        // Prefix is a POD, and are allocated without initializing
        // to prevent redundant work upfront
        // delete[] p;
        free(p);
    }
}

inline void PrefixAllocator::AllocateNextBuffer()
{
    size_t nextSize = m_allocSize == 0 ? 1000 : 2 * m_allocSize;

    // Using malloc here to prevent the ctor of Prefix being called for each item.
    // Instead, the ctor will be called upon first access using GetPrefix
    auto pBuff = reinterpret_cast<Prefix*>(malloc(sizeof(Prefix) * nextSize));

    m_buffers.push_back(pBuff);

    m_allocSize = nextSize;
    m_currOff = 0;
}

template<typename ...Args>
Prefix *PrefixAllocator::GetPrefix(Args&& ...ctorArgs)
{
    if (m_currOff == m_allocSize) {
        AllocateNextBuffer();
    }

    auto buff = m_buffers.back() + m_currOff;

    auto ret = new (buff) Prefix(std::forward<Args>(ctorArgs)...);

    ++m_currOff;

    return ret;
}
