// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

template<typename ...Types>
struct Promote {
	/*
	* The "Promote" object can be used to discover a computation floating type that
	* is both efficient on hardward, and also doesn't result in a loss of precision.
	*
	* Examples:
	* Promote<float>::type == float
	* Promote<double>::type == double
	* Promote<torch::Half>::type == float
	*
	* Additionally, the promote structure can be used to discover the best computation
	* type when given heterogeneous input types.
	*
	* Examples:
	* Promote<float, double>::type == double
	* Promote<float, torch::Half>::type == float
	*/
	typedef float type;
};

template<>
struct Promote<double> { typedef double type; };
template<>
struct Promote<double, double> { typedef double type; };
template<typename A>
struct Promote<A, double> { typedef double type; };
template<typename B>
struct Promote<double, B> { typedef double type; };

template<typename A, typename B, typename ...Rest>
struct Promote<A, B, Rest...> { typedef typename Promote<A, typename Promote<B, Rest...>::type>::type type; };
