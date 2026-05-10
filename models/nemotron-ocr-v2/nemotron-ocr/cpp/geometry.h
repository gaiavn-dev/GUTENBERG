// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>

#ifndef _GEOMETRY_NO_TORCH
#include <torch/torch.h>
#endif

#include "cuda_intellisense.cuh"

#ifndef __NVCC__
#define SORT_ALGO std::sort
#define SWAP std::swap

template<typename ...Args>
using tuple_t = std::tuple<Args...>;

#else

#include <thrust/sort.h>
#include <thrust/tuple.h>

#define SORT_ALGO thrust::sort
#define SWAP thrust::swap

template<typename ...Args>
using tuple_t = thrust::tuple<Args...>;
#endif

template<typename T>
struct Point_ {
    typedef T inner_type;

    T X, Y;

    Point_() = default;

    __any_device__
    Point_(T x, T y) : X(x), Y(y) {}

    __any_device__
    Point_(T *ptr) : X(ptr[0]), Y(ptr[1]) {}

#ifndef _GEOMETRY_NO_TORCH
    template<typename T2>
    __any_device__
    Point_(const torch::TensorAccessor<T2, 1> &accessor) : X(accessor[0]), Y(accessor[1]) {}

    template<typename T2>
    __any_device__
    Point_(const torch::PackedTensorAccessor64<T2, 1> &accessor) : X(accessor[0]), Y(accessor[1]) {}
#endif

    __any_device__
    Point_ &operator+=(const Point_ &other);

    __any_device__
    Point_ &operator-=(const Point_ &other);

    __any_device__
    Point_ &operator*=(const Point_ &other);

    __any_device__
    Point_ &operator/=(const Point_ &other);

    template<typename W>
    __any_device__
    Point_ &operator/=(W w);

    template<typename W>
    __any_device__
    Point_ &operator*=(W w);

    __any_device__
    Point_ operator-() {
        return { -X, -Y };
    }

    __any_device__
    T Sum() const { return X + Y; }

    __any_device__
    T Angle() const;

    __any_device__
    void swap(Point_ &other) noexcept {
        SWAP(X, other.X);
        SWAP(Y, other.Y);
    }
};

template<typename T>
__lib_inline__ __any_device__
void swap(Point_<T> &a, Point_<T> &b) {
    a.swap(b);
}


template<typename T>
__any_device__
__lib_inline__ T Point_<T>::Angle() const {
#ifndef __NVCC__
    using std::atan2;
#endif
    return atan2(Y, X);
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> min(const Point_<T> &a, const Point_<T> &b) {
#ifndef __NVCC__
    using std::min;
#endif
    return {
        min(a.X, b.X),
        min(a.Y, b.Y)
    };
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> max(const Point_<T> &a, const Point_<T> &b) {
#ifndef __NVCC__
    using std::max;
#endif
    return {
        max(a.X, b.X),
        max(a.Y, b.Y)
    };
}

template<typename T>
struct AABB_ {
    typedef T inner_type;

    T X;
    T Y;
    T MaxX;
    T MaxY;

    AABB_() = default;
    __any_device__
    AABB_(T x, T y, T maxX, T maxY)
        : X(x), Y(y), MaxX(maxX), MaxY(maxY) {}

    __any_device__
    bool Contains(const Point_<T> &p) const {
        return p.X >= X && p.X < MaxX &&
               p.Y >= Y && p.Y < MaxY;
    }

    __any_device__ __lib_inline__
    AABB_ Union(const AABB_ &other) const {
#ifndef __NVCC__
        using std::min;
        using std::max;
#endif
        T minX = min(X, other.X);
        T maxX = max(MaxX, other.MaxX);
        T minY = min(Y, other.Y);
        T maxY = max(MaxY, other.MaxY);

        return { minX, minY, maxX, maxY };
    }

    __any_device__
    AABB_ &operator-=(const Point_<T> &offset) {
        X -= offset.X;
        MaxX -= offset.X;
        Y -= offset.Y;
        MaxY -= offset.Y;
        return *this;
    }

    __any_device__
    __lib_inline__ T Width() const { return MaxX - X; }
    __any_device__
    __lib_inline__ T Height() const { return MaxY - Y; }
    __any_device__
    __lib_inline__ T Area() const { return Width() * Height(); }

    __lib_inline__ T &operator[] (int64_t idx)
    {
        static_assert(std::is_standard_layout<AABB_<T>>::value, "This function is only valid for standard layout");
        return (&X)[idx];
    }
    __lib_inline__ T operator[] (int64_t idx) const
    {
        static_assert(std::is_standard_layout<AABB_<T>>::value, "This function is only valid for standard layout");
        return (&X)[idx];
    }

    __any_device__ __lib_inline__
    AABB_ Intersection(const AABB_ &other) const {
#ifndef __NVCC__
        using std::min;
        using std::max;
#endif
        T minX = max(X, other.X);
        T minY = max(Y, other.Y);
        T maxX = min(MaxX, other.MaxX);
        T maxY = min(MaxY, other.MaxY);
        // Prevent negative area
        minX = min(minX, maxX);
        minY = min(minY, maxY);
        return { minX, minY, maxX, maxY };
    }

    __any_device__ __lib_inline__
    T IntersectionArea(const AABB_ &other) const { return Intersection(other).Area(); }
};

template<typename T, typename Derived>
struct QuadBase_ {
    typedef T inner_type;

    __any_device__
    AABB_<T> Bounds() const;

    __any_device__
    bool Contains(const Point_<T> &p) const;

    __any_device__
    T Area() const;

    __any_device__
    T Height() const;

    __any_device__
    T Width() const;

    template<typename Derived2>
    __any_device__
    T IntersectionArea(const QuadBase_<T, Derived2> &other) const;

    template<typename Derived2>
    __any_device__
    T IOU(const QuadBase_<T, Derived2> &other) const;

    template<typename Derived2>
    __any_device__
    T IOU_UpperBound(const QuadBase_<T, Derived2> &other) const;

    __any_device__
    Point_<T> Center() const;

    template<typename Derived2>
    __any_device__
    /*
        Returns 3 geometric associations between the two quads:
            0: The percent shared area between this and other relative to this (e.g. if other contains this, then it returns 1)
            1: The percent shared area between other and this relative to other (e.g. if this contains other, then it return 1)
            2: The IOU of the two quads
    */
    tuple_t<T, T, T> RegionSizes(const QuadBase_<T, Derived2> &other) const;

    template<typename Derived2>
    __any_device__
    tuple_t<T, T, T> RegionSizes_UpperBound(const QuadBase_<T, Derived2> &other) const;

    __any_device__
    Derived &operator/=(T val) {
        auto rcp = 1 / val;
        return *this *= rcp;
    }

    __any_device__
    Derived &operator*=(T val) {
        auto dThis = static_cast<Derived*>(this);
        #pragma unroll
        for (size_t i = 0; i < 4; ++i) {
            dThis->Vertices[i] *= val;
        }
        return *dThis;
    }

    friend auto begin(const QuadBase_ &q) { return static_cast<const Derived&>(q).Vertices; }
    friend auto begin(QuadBase_& q) { return static_cast<const Derived&>(q).Vertices; }
    friend auto end(const QuadBase_ &q) { return static_cast<const Derived&>(q).Vertices + 4; }
    friend auto end(QuadBase_ &q) { return static_cast<const Derived&>(q).Vertices + 4; }
};

template<typename T>
struct Quad_ : QuadBase_<T, Quad_<T>> {
    Point_<T> *Vertices = nullptr;

    Quad_() = default;
    __any_device__
    Quad_(T *dataPtr)
        : Vertices(reinterpret_cast<Point_<T>*>(dataPtr)) {}
    __any_device__
    Quad_(Point_<T> *dataPtr)
        : Vertices(dataPtr) {}

    template<typename index_t>
    __any_device__ __lib_inline__
    const Point_<T> &operator[](index_t offset) const { return Vertices[offset]; }
    template<typename index_t>
    __any_device__ __lib_inline__
    Point_<T> &operator[](index_t offset) { return Vertices[offset]; }
};

template<typename T>
struct InPlaceQuad_ : public QuadBase_<T, InPlaceQuad_<T>> {
    Point_<T> Vertices[4];

    InPlaceQuad_() = default;
    __any_device__
    InPlaceQuad_(const T *dataPtr)
    {
#if defined(__NVCC__)
        T *pVals = reinterpret_cast<T*>(Vertices);
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            pVals[i] = dataPtr[i];
        }
#else
        using std::copy;
        copy(dataPtr, dataPtr + 8, reinterpret_cast<T*>(Vertices));
#endif
    }
    __any_device__
    InPlaceQuad_(const Point_<T> *dataPtr)
    {
#if defined(__NVCC__)
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            Vertices[i] = dataPtr[i];
        }
#else
        using std::copy;
        copy(dataPtr, dataPtr + 4, Vertices);
#endif
    }

    template<typename index_t>
    __any_device__ __lib_inline__
    const Point_<T> &operator[](index_t v) const { return Vertices[v]; }

    template<typename index_t>
    __any_device__ __lib_inline__
    Point_<T> &operator[](index_t v) { return Vertices[v]; }
};

template<typename T, typename Derived>
struct PolygonBase_ {
    typedef T inner_type;

    __any_device__
    AABB_<T> Bounds() const;

    __any_device__
    bool Contains(const Point_<T> &p) const;

    __any_device__
    T EdgeLength() const;

    __any_device__
    Point_<T> Center() const;

    __any_device__
    T Area() const;
};

template<typename T>
struct Polygon_ : PolygonBase_<T, Polygon_<T>> {
    Point_<T> *Vertices = nullptr;
    size_t Count = 0;

    Polygon_() = default;
    __any_device__
    Polygon_(T *dataPtr, size_t vertexCount)
        : Vertices(reinterpret_cast<Point_<T>*>(dataPtr)), Count(vertexCount) {}
    __any_device__
    Polygon_(Point_<T> *dataPtr, size_t vertexCount)
        : Vertices(dataPtr), Count(vertexCount) {}

    __any_device__
    const Point_<T> &operator[](size_t offset) const { return Vertices[offset]; }
    __any_device__
    Point_<T> &operator[](size_t offset) { return Vertices[offset]; }
};

template<typename T>
struct Segment_ {
    Point_<T> A, B;

    Segment_() = default;
    __any_device__
    Segment_(const Point_<T> &a, const Point_<T> &b) : A(a), B(b) {}

    __any_device__
    T Length() const;
    __any_device__
    T LengthSq() const;
    __any_device__
    bool Intersection(const Segment_<T> &other, Point_<T> &out_ptAlong) const;
};

template<typename T>
__any_device__
__lib_inline__ Point_<T> operator+(const Point_<T> &a, const Point_<T> &b) {
    return { a.X + b.X, a.Y + b.Y };
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> operator-(const Point_<T> &a, const Point_<T> &b) {
    return { a.X - b.X, a.Y - b.Y };
}

template<typename T, typename W>
__any_device__
__lib_inline__ Point_<T> operator*(W scale, const Point_<T> &p) {
    return { scale * p.X, scale * p.Y };
}

template<typename T, typename W>
__any_device__
__lib_inline__ Point_<T> operator*(const Point_<T> &p, W scale) {
    return { scale * p.X, scale * p.Y };
}

template<typename T, typename W>
__any_device__
__lib_inline__ Point_<T> operator/(const Point_<T> &p, W divisor) {
    return { p.X / divisor, p.Y / divisor };
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> operator*(const Point_<T> &a, const Point_<T> &b) {
    return { a.X * b.X, a.Y * b.Y };
}

template<typename T, typename W>
__any_device__
__lib_inline__ Point_<T> operator-(const Point_<T> &p, W v) {
    return { p.X - v, p.Y - v };
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> &Point_<T>::operator+=(const Point_<T> &p) {
    X = X + p.X;
    Y = Y + p.Y;
    return *this;
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> &Point_<T>::operator-=(const Point_<T> &p) {
    X = X - p.X;
    Y = Y - p.Y;
    return *this;
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> &Point_<T>::operator*=(const Point_<T> &p) {
    X = X * p.X;
    Y = Y * p.Y;
    return *this;
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> &Point_<T>::operator/=(const Point_<T> &p) {
    X = X / p.X;
    Y = Y / p.Y;
    return *this;
}

template<typename T>
template<typename W>
__any_device__
__lib_inline__ Point_<T> &Point_<T>::operator/=(W val) {
    // TODO: This can be more efficient for float types by computing the reciprocal
    X /= val;
    Y /= val;
    return *this;
}

template<typename T>
template<typename W>
__any_device__
__lib_inline__ Point_<T> &Point_<T>::operator*=(W val) {
    X *= val;
    Y *= val;
    return *this;
}

template<typename T>
__any_device__
__lib_inline__ T dot(const Point_<T> &a, const Point_<T> &b) {
    return a.X * b.X + a.Y * b.Y;
}

template<typename T>
__any_device__
__lib_inline__ T dot(const Point_<T> &p) {
    return dot(p, p);
}

template<typename T>
__any_device__
__lib_inline__ T length(const Point_<T> &p) {
#ifndef __NVCC__
    using std::sqrt;
#endif
    return sqrt(dot(p));
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> normalize(const Point_<T> &p) {
    static constexpr T epsilon = std::numeric_limits<T>::epsilon();
    auto len = length(p) + epsilon;
    return { p.X / len, p.Y / len };
}

template<typename T>
__any_device__
__lib_inline__ Point_<T> ortho_2d(const Point_<T> &p) {
    return { -p.Y, p.X };
}

template<typename T>
__host__
__lib_inline__ std::ostream &operator<<(std::ostream &os, const Point_<T> &p) {
    return os << "(" << p.X << ", " << p.Y << ")";
}

template<typename T>
__host__
__lib_inline__ std::ostream &operator<<(std::ostream &os, const AABB_<T> &b) {
    return os << "[(" << b.X << ", " << b.Y << "), (" << b.MaxX << ", " << b.MaxY << ")]";
}

template<typename T>
__host__
__lib_inline__ std::ostream &operator<<(std::ostream &os, const Segment_<T> &s) {
    return os << "[(" << s.A.X << ", " << s.A.Y << "), (" << s.B.X << ", " << s.B.Y << ")]";
}

template<typename T>
__host__
__lib_inline__ std::ostream &operator<<(std::ostream &os, const Quad_<T> &q) {
    os << "[" << q.Vertices[0];
    for (size_t i = 1; i < 4; ++i) {
        os << ", " << q.Vertices[i];
    }
    return os << "]";
}

template<typename T>
__any_device__
__lib_inline__ int _signum(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename T>
__any_device__
__lib_inline__ T sign(const Point_<T> &p1, const Point_<T> &p2, const Point_<T> &p3) {
    T ret = (p1.X - p3.X) * (p2.Y - p3.Y) - (p2.X - p3.X) * (p1.Y - p3.Y);
    auto sgn = _signum(ret);
    return sgn;
}

template<typename T>
__any_device__
__lib_inline__ T Segment_<T>::Length() const
{
#ifndef __NVCC__
    using std::sqrt;
#endif
    return sqrt(LengthSq());
}

template<typename T>
__any_device__
__lib_inline__ T Segment_<T>::LengthSq() const
{
    return dot(B - A);
}

template<typename T>
__any_device__
inline bool Segment_<T>::Intersection(const Segment_<T> &other, Point_<T> &out_ptAlong) const
{
    auto p1 = A, p2 = B, p3 = other.A, p4 = other.B;

    auto denom = (p4.Y - p3.Y) * (p2.X - p1.X) - (p4.X - p3.X) * (p2.Y - p1.Y);

    if (abs(denom) < 1e-8) {
        return false;
    }

    auto numer = (p4.X - p3.X) * (p1.Y - p3.Y) - (p4.Y - p3.Y) * (p1.X - p3.X);

    auto t = numer / denom;

    auto Bnumer = (p2.X - p1.X) * (p1.Y - p3.Y) - (p2.Y - p1.Y) * (p1.X - p3.X);

    auto Bt = Bnumer / denom;

    if (t < 0 || t > 1 || Bt < 0 || Bt > 1) {
        return false;
    }

    out_ptAlong = A + t * (B - A);

    return true;
}

template<typename quad_t>
__any_device__
auto quad_center(const quad_t &quad) -> Point_<typename quad_t::inner_type>
{
    typedef typename quad_t::inner_type T;

    Point_<T> center = quad[0];
    for (size_t i = 1; i < 4; ++i) {
        center += quad[i];
    }

    return center / T{ 4 };
}

template<typename T, typename Derived>
__any_device__
Point_<T> QuadBase_<T, Derived>::Center() const {
    return quad_center(static_cast<const Derived&>(*this));
}

template<typename quad_t>
__any_device__
auto quad_bounds(const quad_t &quad) -> AABB_<typename quad_t::inner_type>
{
#ifndef __NVCC__
    using std::min;
    using std::max;
#endif
    auto minP = quad[0];
    auto maxP = minP;
    for (size_t i = 1; i < 4; ++i) {
        auto qp = quad[i];
        minP = min(minP, qp);
        maxP = max(maxP, qp);
    }
    return { minP.X, minP.Y, maxP.X, maxP.Y };
}

template<typename T, typename Derived>
__any_device__
AABB_<T> QuadBase_<T, Derived>::Bounds() const {
    return quad_bounds(static_cast<const Derived&>(*this));
}

template<typename Quad_t, typename point_t>
__any_device__
inline bool quad_contains(const Quad_t &quad, const point_t &pt)
{
#ifndef __NVCC__
    using std::abs;
#endif

    // Checks that the point lies on the interior side of each half plane
    auto d1 = sign(pt, quad[0], quad[1]);
    auto d2 = sign(pt, quad[1], quad[2]);
    auto d3 = sign(pt, quad[2], quad[3]);
    auto d4 = sign(pt, quad[3], quad[0]);

    // bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0) || (d4 < 0);
    // bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0) || (d4 > 0);
    int tot = d1 + d2 + d3 + d4;

    // return !(has_neg && has_pos);
    return abs(tot) == 4;
}

template<typename T, typename Derived>
__any_device__
__lib_inline__ bool QuadBase_<T, Derived>::Contains(const Point_<T> &pt) const
{
    return quad_contains(static_cast<const Derived&>(*this), pt);
}

template<typename PtList>
__any_device__
inline auto shoelace_area(const PtList &points, size_t numPts, bool isSigned=false) -> decltype(points[0].X)
{
#ifndef __NVCC__
    using std::abs;
#endif

    decltype(points[0].X) area = 0;

    size_t j = numPts - 1;
    for (size_t i = 0; i < numPts; ++i) {
        auto Pi = points[i];
        auto Pj = points[j];

        area += (Pj.X + Pi.X) * (Pj.Y - Pi.Y);
        j = i;
    }

    area = area / 2;

    if (! isSigned) {
        area = abs(area);
    }

    return area;
}

template<typename T, typename Derived>
__any_device__
__lib_inline__ T QuadBase_<T, Derived>::Height() const
{
    auto &d = static_cast<const Derived&>(*this);
    auto h1 = Segment_<T>(d[1], d[2]).Length();
    auto h2 = Segment_<T>(d[3], d[0]).Length();
    return (h1 + h2) / 2;
}

template<typename T, typename Derived>
__any_device__
__lib_inline__ T QuadBase_<T, Derived>::Width() const
{
    auto &d = static_cast<const Derived&>(*this);
    auto w1 = Segment_<T>(d[0], d[1]).Length();
    auto w2 = Segment_<T>(d[3], d[2]).Length();
    return (w1 + w2) / 2;
}

// A quad can be defined as the sum of the area of two triangles
template<typename T, typename Derived>
__any_device__
inline T QuadBase_<T, Derived>::Area() const
{
    // auto vertices = static_cast<const Derived *>(this)->Vertices;
    return shoelace_area(static_cast<const Derived&>(*this), 4);
}

template<typename Quad_t1, typename Quad_t2>
__any_device__
inline auto intersection_area(const Quad_t1 &quadsA, const Quad_t2 &quadsB) -> typename Quad_t1::inner_type
{
#ifndef __NVCC__
    using std::atan2;
#endif

    typedef typename Quad_t1::inner_type T;

    static const size_t MAX_PTS = 32;

    Point_<T> points[MAX_PTS], sortedPoints[MAX_PTS];
    T angles[MAX_PTS];
    size_t indices[MAX_PTS];
    size_t numPts = 0;

    auto addPt = [&] (const Point_<T> &p) {
        points[numPts] = p;
        ++numPts;
    };

    for (size_t i = 0; i < 4; ++i) {
        Point_<T> aPt = quadsA[i];
        Point_<T> bPt = quadsB[i];

        if (quadsA.Contains(bPt)) {
            addPt(bPt);
        }
        if (quadsB.Contains(aPt)) {
            addPt(aPt);
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        Segment_<T> segA{ quadsA[i], quadsA[(i + 1) % 4] };

        for (size_t j = 0; j < 4; ++j) {
            Segment_<T> segB{ quadsB[j], quadsB[(j + 1) % 4] };

            Point_<T> ptAlong;
            if (segA.Intersection(segB, ptAlong)) {
                addPt(ptAlong);
            }
        }
    }

    if (numPts == 0) {
        return 0;
    }

    Point_<T> center{ 0, 0 };
    for (size_t i = 0; i < numPts; ++i) {
        center += points[i];
    }
    center /= numPts;

    for (size_t i = 0; i < numPts; ++i) {
        points[i] -= center;

        angles[i] = atan2(points[i].Y, points[i].X);

        indices[i] = i;
    }

    // Perform an argsort over the angles
    SORT_ALGO(indices, indices + numPts,
        [&] (size_t a, size_t b) {
            return angles[a] < angles[b];
        }
    );

    for (size_t i = 0; i < numPts; ++i) {
        sortedPoints[i] = points[indices[i]];
    }

    // Finally, we can compute the area of this polygon using the shoelace formula
    T area = shoelace_area(sortedPoints, numPts);

    return area;
}

template<typename T, typename Derived>
template<typename Derived2>
__any_device__
__lib_inline__ T QuadBase_<T, Derived>::IntersectionArea(const QuadBase_<T, Derived2> &other) const
{
    return intersection_area(
        static_cast<const Derived&>(*this),
        static_cast<const Derived2&>(other)
    );
}

template<typename T1, typename T2>
__any_device__
__lib_inline__ auto geometry_iou(const T1 &a, const T2 &b) -> decltype(a.Area())
{
    auto aArea = a.Area();
    auto bArea = b.Area();
    auto ixArea = a.IntersectionArea(b);

    auto unionArea = aArea + bArea - ixArea;

    return ixArea / unionArea;
}

template<typename T, typename Derived>
template<typename Derived2>
__any_device__
__lib_inline__ T QuadBase_<T, Derived>::IOU(const QuadBase_<T, Derived2> &other) const
{
    return geometry_iou(
        static_cast<const Derived&>(*this),
        static_cast<const Derived2&>(other)
    );
}

template<typename T, typename Derived>
template<typename Derived2>
__any_device__
__lib_inline__ T QuadBase_<T, Derived>::IOU_UpperBound(const QuadBase_<T, Derived2> &other) const
{
    return geometry_iou(
        Bounds(),
        other.Bounds()
    );
}

template<typename T1, typename T2>
__any_device__ __lib_inline__
auto geometry_region_sizes(const T1 &a, const T2 &b) -> tuple_t<decltype(a.Area()), decltype(a.Area()), decltype(a.IntersectionArea(b))>
{
    auto aArea = a.Area();
    auto bArea = b.Area();
    auto ixArea = a.IntersectionArea(b);

    auto unionArea = aArea + bArea - ixArea;
    auto iou = ixArea / unionArea;

    return { ixArea / aArea, ixArea / bArea, iou };
}


template<typename T, typename Derived>
template<typename Derived2>
__any_device__ __lib_inline__
tuple_t<T, T, T> QuadBase_<T, Derived>::RegionSizes(const QuadBase_<T, Derived2> &other) const
{
    return geometry_region_sizes(
        static_cast<const Derived&>(*this),
        static_cast<const Derived2&>(other)
    );
}

template<typename T, typename Derived>
template<typename Derived2>
__any_device__ __lib_inline__
tuple_t<T, T, T> QuadBase_<T, Derived>::RegionSizes_UpperBound(const QuadBase_<T, Derived2> &other) const
{
    return geometry_region_sizes(
        Bounds(),
        other.Bounds()
    );
}

template<typename polygon_t>
__any_device__
auto polygon_bounds(const polygon_t &poly) -> AABB_<typename polygon_t::inner_type>
{
#ifndef __NVCC__
    using std::min;
    using std::max;
#endif
    auto minP = poly[0];
    auto maxP = minP;
    for (size_t i = 1; i < poly.Count; ++i) {
        auto qp = poly[i];
        minP = min(minP, qp);
        maxP = max(maxP, qp);
    }
    return { minP.X, minP.Y, maxP.X, maxP.Y };
}

template<typename T, typename Derived>
__any_device__
AABB_<T> PolygonBase_<T, Derived>::Bounds() const {
    return polygon_bounds(static_cast<const Derived&>(*this));
}

template<typename polygon_t, typename point_t>
__any_device__
bool polygon_contains(const polygon_t &poly, const point_t &pt)
{
    typedef typename polygon_t::inner_type T;

    // Some arbitrary segment. Technically this should be a ray, but functionally this will work
    Segment_<T> testSeg{ pt, { -1e6, -2e6 }};
    Point_<T> trash;

    int32_t ixCount = 0;
    for (size_t i = 0; i < poly.Count; ++i) {
        Segment_<T> polySeg{ poly[i], poly[(i + 1) % poly.Count] };

        if (testSeg.Intersection(polySeg, trash)) {
            ++ixCount;
        }
    }

    // If there are an odd number of intersections, then the point is inside
    return (ixCount % 2) == 1;
}

template<typename T, typename Derived>
__any_device__
bool PolygonBase_<T, Derived>::Contains(const Point_<T> &pt) const {
    return polygon_contains(static_cast<const Derived&>(*this), pt);
}

template<typename polygon_t>
__any_device__
auto polygon_edge_length(const polygon_t &poly) -> typename polygon_t::inner_type
{
    typedef typename polygon_t::inner_type T;

    T ret = 0;

    for (size_t i = 0; i < poly.Count; ++i) {
        Segment_<T> seg{ poly[i], poly[(i + 1) % poly.Count] };

        ret += seg.Length();
    }

    return ret;
}

template<typename T, typename Derived>
__any_device__
T PolygonBase_<T, Derived>::EdgeLength() const {
    return polygon_edge_length(static_cast<const Derived&>(*this));
}

template<typename polygon_t>
__any_device__
auto polygon_center(const polygon_t &poly) -> Point_<typename polygon_t::inner_type>
{
    typedef typename polygon_t::inner_type T;

    T cx = 0, cy = 0, a = 0;
    size_t j = poly.Count - 1;
    for (size_t i = 0; i < poly.Count; ++i) {
        Point_<T> p0 = poly[i];
        Point_<T> p1 = poly[j];

        T common = (p0.X * p1.Y - p1.X * p0.Y);
        cx += (p0.X + p1.X) * common;
        cy += (p0.Y + p1.Y) * common;
        a += common;

        j = i;
    }

    a /= 2;

    Point_<T> center{ cx / (6 * a), cy / (6 * a) };

    return center;
}

template<typename T, typename Derived>
__any_device__
Point_<T> PolygonBase_<T, Derived>::Center() const {
    return polygon_center(static_cast<const Derived&>(*this));
}

template<typename T, typename Derived>
__any_device__
T PolygonBase_<T, Derived>::Area() const {
    const Derived &dThis = static_cast<const Derived&>(*this);
    return shoelace_area(dThis, dThis.Count);
}


template<typename T>
__any_device__
Point_<T> nearest_point_on_segment(const Point_<T> &pt, const Segment_<T> &seg)
{
#ifndef __NVCC__
    using std::max;
    using std::min;
#endif

    const T l2 = seg.LengthSq();

    if (l2 == 0.0) {
        return seg.A;
    }

    const auto v = seg.A;
    const auto w = seg.B;
    // Consider the line extending the segment, parameterized as v + t*(w-v)
    // Find projection of point p onto the line
    auto t = dot(pt - v, w - v) / l2;

    // Clamp between t=0 and t=1
    t = max(static_cast<T>(0), min(static_cast<T>(1), t));

    const auto projection = v + t * (w - v);

    return projection;
}


template<typename T>
__any_device__
Segment_<T> shortest_line_between_segments(const Segment_<T> &a, const Segment_<T> &b)
{
    Segment_<T> segs[] = {
        { a.A, nearest_point_on_segment(a.A, b) },
        { a.B, nearest_point_on_segment(a.B, b) },
        { nearest_point_on_segment(b.A, a), b.A },
        { nearest_point_on_segment(b.B, a), b.B }
    };

    T minDist = std::numeric_limits<T>::max();
    size_t idx;

    #pragma unroll
    for (size_t i = 0; i < 4; ++i) {
        T dist = segs[i].LengthSq();
        if (dist < minDist) {
            minDist = dist;
            idx = i;
        }
    }

    return segs[idx];
}

// Find the distance between a point and the nearest point along the specified segment
template<typename T>
__any_device__
T distance_to_segment(const Point_<T> &pt, const Segment_<T> &seg)
{
    auto projection = nearest_point_on_segment(pt, seg);

    auto dist = length(pt - projection);

    return dist;
}
