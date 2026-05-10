// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <stack>

#include "../geometry.h"


#define MODE_GEOMETRY 0x02ull
#define MODE_CHILDREN 0x00ull

#define DIM_X 0x0ull
#define DIM_Y 0x1ull

static const size_t INVALID_IDX = -1;

template<typename T>
struct NMS_BoundsWrapper
{
    typedef std::unique_ptr<NMS_BoundsWrapper> Ptr;
    typedef AABB_<typename T::inner_type> bds_t;

    size_t GeoIdx;
    const T *Geometry;
    bds_t Bounds;

    NMS_BoundsWrapper(size_t geoIdx, const T *geometry) : GeoIdx(geoIdx), Geometry(geometry), Bounds(geometry->Bounds()) { }
};

template<typename T>
class NMS_NodeAllocator;

template<typename T>
class NMS_KDTree;

template<typename T>
class NMS_BuildCache;

template<typename T>
class NMS_KDNode
{
    friend class NMS_KDTree<T>;

public:
    typedef NMS_BoundsWrapper<T> bds_t;
    typedef std::unique_ptr<NMS_KDNode[]> UPtr;
    typedef typename T::inner_type inner_type;
    typedef std::vector<bds_t*> geo_vec_t;
    typedef std::unique_ptr<geo_vec_t> geo_vec_ptr;

    void Build(geo_vec_ptr geometries, const typename bds_t::bds_t &envelope,
               NMS_NodeAllocator<T> &allocator, NMS_BuildCache<T> &buildCache);

    template<typename Fn>
    void FindIntersections(size_t geoIdx, const typename bds_t::bds_t &bds, const Fn &fn) const;

private:
    inline uintptr_t Dim() const { return reinterpret_cast<uintptr_t>(m_ptr) & 0x01ull; }
    inline uintptr_t Mode() const { return reinterpret_cast<uintptr_t>(m_ptr) & 0x02ull; }
    inline void Children(NMS_KDNode *&children, inner_type &splitPos) const
    {
        auto vPtr = Geometries();
        splitPos = *reinterpret_cast<inner_type*>(vPtr);
        children = reinterpret_cast<NMS_KDNode*>(vPtr + sizeof(inner_type));
    }
    inline uint8_t* Geometries() const
    {
        return reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(m_ptr) & ~0x3ull);
    }
    inline void SetPtr(uint8_t *vPtr, uintptr_t mode, uintptr_t dim)
    {
        m_ptr = reinterpret_cast<uint8_t*>(
            reinterpret_cast<uintptr_t>(vPtr) | mode | dim
        );
    }
    void AssignGeometries(geo_vec_ptr geometries, NMS_BuildCache<T> &buildCache);

    uint8_t *m_ptr;
};

template<typename T>
class NMS_NodeAllocator
{
public:
    typedef NMS_KDNode<T> node_t;
    typedef typename node_t::inner_type inner_type;

    NMS_NodeAllocator(size_t initialGuess = 512);
    ~NMS_NodeAllocator();

    void Get(size_t numNodes, NMS_KDNode<T> *&outNodes, inner_type *&outSplitPos, uint8_t *&outRawPtr);

private:
    std::vector<std::pair<size_t, uint8_t*>> m_buffers;
    size_t m_offset;
};

template<typename T>
class NMS_BuildCache
{
public:
    typedef typename NMS_KDNode<T>::bds_t bds_t;
    typedef std::unique_ptr<NMS_BuildCache> Ptr;
    typedef std::vector<bds_t*> geo_vec_t;
    typedef std::unique_ptr<geo_vec_t> geo_vec_ptr;

    NMS_BuildCache(size_t initialSize);
    ~NMS_BuildCache();

    geo_vec_ptr Get(size_t sizeHint);
    bds_t** GetRawBuffer(size_t numGeos, uint8_t *&rawPtr);

    void Release(geo_vec_ptr buff);

private:
    std::stack<geo_vec_ptr> m_cache;
    std::vector<std::pair<size_t, uint8_t*>> m_rawBuffers;
    size_t m_rawOffset;
};


template<typename T>
class NMS_KDTree
{
    typedef typename T::inner_type inner_type;
    typedef NMS_BoundsWrapper<T> bds_t;
    typedef NMS_KDNode<T> node_t;

public:
    NMS_KDTree();
    ~NMS_KDTree();

    void Build(const std::vector<T> &geometries);

    template<typename Fn>
    void FindIntersections(size_t geoIdx, const Fn &fn) const;

    template<typename Fn>
    void FindIntersections(const T &geo, const Fn &fn) const;

private:
    bds_t *m_wrappers;
    NMS_NodeAllocator<T> m_allocator;
    node_t m_root;
    typename NMS_BuildCache<T>::Ptr m_buildCache;
};

template<typename T>
NMS_KDTree<T>::NMS_KDTree()
    : m_wrappers(nullptr)
{
    m_root.m_ptr = nullptr;
}

template<typename T>
NMS_KDTree<T>::~NMS_KDTree()
{
    free(m_wrappers);
}

template<typename T>
void NMS_KDTree<T>::Build(const std::vector<T> &geometries)
{
    if (geometries.empty()) {
        m_root.m_ptr = nullptr;
        return;
    }

    // Doing this so that we can perform placement-new on the array buffer, and thus
    // can only perform a single memory allocation for all geometries at once
    m_wrappers = reinterpret_cast<bds_t*>(malloc(sizeof(bds_t) * geometries.size()));

    m_buildCache.reset(new NMS_BuildCache<T>(geometries.size()));

    auto bdsGeos = m_buildCache->Get(geometries.size());

    typename bds_t::bds_t envelope;

    for (size_t i = 0; i < geometries.size(); ++i) {
        // Placement new. Constructs the object in the place specified in the first (...)
        new (m_wrappers + i) bds_t(i, &geometries[i]);

        bdsGeos->push_back(m_wrappers + i);
        if (i == 0) {
            envelope = m_wrappers[i].Bounds;
        } else {
            envelope = envelope.Union(m_wrappers[i].Bounds);
        }
    }


    m_root.Build(std::move(bdsGeos), envelope, m_allocator, *m_buildCache);
}

template<typename T>
void NMS_KDNode<T>::Build(geo_vec_ptr geometries, const typename bds_t::bds_t &envelope,
                          NMS_NodeAllocator<T> &allocator, NMS_BuildCache<T> &buildCache)
{
    static const size_t MAX_GEOMETRIES = 8;

    if (geometries->size() <= MAX_GEOMETRIES) {
        AssignGeometries(std::move(geometries), buildCache);
    } else {
        geo_vec_ptr leftGeos = buildCache.Get(geometries->size()),
                    rightGeos = buildCache.Get(geometries->size());

        inner_type szX = envelope[2] - envelope[0];
        inner_type szY = envelope[3] - envelope[1];

        int64_t dim = szX > szY ? 0 : 1;
        auto emn = envelope[dim];
        auto emx = envelope[dim + 2];

        auto pivotPos = (emn + emx) / 2;
        for (bds_t *g : *geometries) {
            auto mn = g->Bounds[dim];
            auto mx = g->Bounds[dim + 2];

            if (mn < pivotPos) {
                leftGeos->push_back(g);
            }
            if (mx > pivotPos) {
                rightGeos->push_back(g);
            }
        }

        if (leftGeos->size() == geometries->size() || rightGeos->size() == geometries->size()) {
            AssignGeometries(std::move(geometries), buildCache);
            buildCache.Release(std::move(leftGeos));
            buildCache.Release(std::move(rightGeos));
        } else {
            buildCache.Release(std::move(geometries));

            inner_type *nodeSplitPos;
            uint8_t *nodeRawPtr;
            NMS_KDNode *children;
            allocator.Get(2, children, nodeSplitPos, nodeRawPtr);

            SetPtr(nodeRawPtr, MODE_CHILDREN, dim);
            *nodeSplitPos = pivotPos;

            typename bds_t::bds_t leftEnv{envelope}, rightEnv{envelope};
            // Set the max of the left envelope to the split plane
            leftEnv[dim + 2] = pivotPos;
            // Set the min of the right envelope to the split plane
            rightEnv[dim] = pivotPos;

            children[0].Build(std::move(leftGeos), leftEnv, allocator, buildCache);
            children[1].Build(std::move(rightGeos), rightEnv, allocator, buildCache);
        }
    }
}

template<typename T>
void NMS_KDNode<T>::AssignGeometries(geo_vec_ptr geometries, NMS_BuildCache<T> &buildCache)
{
    if (geometries->empty()) {
        SetPtr(nullptr, MODE_GEOMETRY, 0);
    } else {
        uint8_t *vPtr;
        bds_t **geoPtr = buildCache.GetRawBuffer(geometries->size(), vPtr);
        std::copy(geometries->begin(), geometries->end(), geoPtr);

        SetPtr(vPtr, MODE_GEOMETRY, 0);
    }
    buildCache.Release(std::move(geometries));
}

template<typename T>
template<typename Fn>
void NMS_KDTree<T>::FindIntersections(size_t geoIdx, const Fn &fn) const
{
    if (!m_wrappers) return;

    auto &bds = m_wrappers[geoIdx].Bounds;

    m_root.FindIntersections(geoIdx, bds, fn);
}

template<typename T>
template<typename Fn>
void NMS_KDTree<T>::FindIntersections(const T &geo, const Fn &fn) const
{
    if (!m_wrappers) return;

    NMS_BoundsWrapper<T> bdsWrapper(INVALID_IDX, &geo);

    m_root.FindIntersections(INVALID_IDX, bdsWrapper.Bounds, fn);
}

template<typename T>
template<typename Fn>
void NMS_KDNode<T>::FindIntersections(size_t geoIdx, const typename bds_t::bds_t &bds, const Fn &fn) const
{
    auto mode = Mode();

    if (mode == MODE_GEOMETRY) {
        auto *vPtr = Geometries();

        size_t numGeos = *reinterpret_cast<size_t*>(vPtr);
        bds_t **geoPtr = reinterpret_cast<bds_t**>(vPtr + sizeof(size_t));

        bds_t **endPtr = geoPtr + numGeos;
        for (; geoPtr != endPtr; ++geoPtr) {
            const bds_t *child = *geoPtr;

            // Don't compute this against self
            if (geoIdx != INVALID_IDX && child->GeoIdx <= geoIdx) continue;

            typename bds_t::bds_t::inner_type pctN, pctM, iou;
            std::tie(pctN, pctM, iou) = geometry_region_sizes(bds, child->Bounds);

            if (iou > 0) {
                fn(child->GeoIdx, pctN, pctM, iou);
            }
        }
    } else {
        auto dim = Dim();

        auto mn = bds[dim];
        auto mx = bds[dim + 2];

        NMS_KDNode *children;
        inner_type splitPos;
        Children(children, splitPos);

        if (mn < splitPos) {
            children[0].FindIntersections(geoIdx, bds, fn);
        }
        if (mx > splitPos) {
            children[1].FindIntersections(geoIdx, bds, fn);
        }
    }
}

template<typename T>
NMS_NodeAllocator<T>::NMS_NodeAllocator(size_t initialGuess)
    : m_offset(0)
{
    size_t allocSize = initialGuess * (sizeof(inner_type) + 2 * sizeof(node_t));
    auto ptr = reinterpret_cast<uint8_t*>(malloc(allocSize));
    m_buffers.emplace_back(initialGuess, ptr);
}

template<typename T>
NMS_NodeAllocator<T>::~NMS_NodeAllocator()
{
    for (auto &p : m_buffers) {
        free(p.second);
    }
}

template<typename T>
void NMS_NodeAllocator<T>::Get(size_t numNodes, node_t *&outNodes, inner_type *&outSplitPos, uint8_t *&outRawPtr)
{
    auto &currBuff = m_buffers.back();

    size_t rem = currBuff.first - m_offset;

    size_t reqSize = sizeof(inner_type) + sizeof(node_t) * numNodes;

    if (rem >= reqSize) {
        outRawPtr = currBuff.second + m_offset;
        outSplitPos = reinterpret_cast<inner_type*>(outRawPtr);
        outNodes = reinterpret_cast<node_t*>(outRawPtr + sizeof(inner_type));
        m_offset += reqSize;
        return;
    }

    // Rounds up to the nearest factor of 2
    size_t allocSize = (std::max(currBuff.first * 2, reqSize) + 1) & ~0x01ull;
    auto ptr = reinterpret_cast<uint8_t*>(malloc(allocSize));
    m_buffers.emplace_back(allocSize, ptr);
    m_offset = 0;

    Get(numNodes, outNodes, outSplitPos, outRawPtr);
}

template<typename T>
NMS_BuildCache<T>::NMS_BuildCache(size_t initialSize)
    : m_rawOffset(0)
{
    auto allocSize = sizeof(bds_t*) * initialSize * 2;
    auto raw1 = reinterpret_cast<uint8_t*>(malloc(allocSize));
    m_rawBuffers.emplace_back(allocSize, raw1);
}

template<typename T>
NMS_BuildCache<T>::~NMS_BuildCache()
{
    for (auto &p : m_rawBuffers) {
        free(p.second);
    }
}

template<typename T>
typename NMS_BuildCache<T>::geo_vec_ptr NMS_BuildCache<T>::Get(size_t sizeHint)
{
    geo_vec_ptr ret;
    if (! m_cache.empty()) {
        ret = std::move(m_cache.top());
        m_cache.pop();
        ret->clear();
    } else {
        ret.reset(new std::vector<bds_t*>);
    }

    ret->reserve(sizeHint);
    return ret;
}

template<typename T>
typename NMS_BuildCache<T>::bds_t** NMS_BuildCache<T>::GetRawBuffer(size_t numGeos, uint8_t *&rawPtr)
{
    auto &currBuff = m_rawBuffers.back();
    size_t rem = currBuff.first - m_rawOffset;

    size_t reqSize = sizeof(size_t) + sizeof(bds_t*) * numGeos;

    if (rem >= reqSize) {
        rawPtr = currBuff.second + m_rawOffset;
        m_rawOffset += reqSize;
        reinterpret_cast<size_t*>(rawPtr)[0] = numGeos;
        return reinterpret_cast<bds_t**>(rawPtr + sizeof(size_t));
    }

    size_t allocSize = (std::max(currBuff.first * 2, reqSize) + 1) & ~0x01ull;
    auto ptr = reinterpret_cast<uint8_t*>(malloc(allocSize));
    m_rawBuffers.emplace_back(allocSize, ptr);
    m_rawOffset = 0;

    return GetRawBuffer(numGeos, rawPtr);
}

template<typename T>
void NMS_BuildCache<T>::Release(geo_vec_ptr buff)
{
    m_cache.push(std::move(buff));
}

#undef MODE_GEOMETRY
#undef MODE_CHILDREN
#undef DIM_X
#undef DIM_Y
