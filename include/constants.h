/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Type definitions and constants for HPDBSCAN
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef CONSTANTS_H
#define	CONSTANTS_H

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <vector>

using ssize_t       = ptrdiff_t;
using Locator       = std::pair<size_t, size_t>;
using Cluster       = ssize_t;
using Clusters      = std::vector<Cluster>;
using Cell          = size_t;
using Cells         = std::vector<Cell>;
using CellHistogram = std::map<Cell, size_t>;
using CellIndex     = std::map<Cell, Locator>;
using CellBounds    = std::array<size_t, 4>;
using ComputeBounds = std::array<size_t, 2>;
using Cuts          = std::vector<Locator>;

static const size_t  BITS_PER_BYTE = 8;
static const ssize_t NOT_VISITED   = std::numeric_limits<ssize_t>::max();
static const ssize_t NOISE         = std::numeric_limits<ssize_t>::max() - 1;

const std::vector<size_t> RADIX_POWERS = {
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        10000000000,
        100000000000,
        1000000000000,
        10000000000000,
        100000000000000,
        1000000000000000,
        10000000000000000,
        100000000000000000,
        1000000000000000000,
};
const size_t RADIX_BUCKETS = 10; // radix sort buckets

#endif // CONSTANTS_H
