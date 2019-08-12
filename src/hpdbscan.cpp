/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Highly parallel DBSCAN algorithm implementation
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#include <cstdint>

#include "hpdbscan.h"

// explicit template instantiation
template Clusters HPDBSCAN::cluster<uint8_t >(Dataset&, int);
template Clusters HPDBSCAN::cluster<uint16_t>(Dataset&, int);
template Clusters HPDBSCAN::cluster<uint32_t>(Dataset&, int);
template Clusters HPDBSCAN::cluster<uint64_t>(Dataset&, int);

template Clusters HPDBSCAN::cluster<int8_t >(Dataset&, int);
template Clusters HPDBSCAN::cluster<int16_t>(Dataset&, int);
template Clusters HPDBSCAN::cluster<int32_t>(Dataset&, int);
template Clusters HPDBSCAN::cluster<int64_t>(Dataset&, int);

template Clusters HPDBSCAN::cluster<float >(Dataset&, int);
template Clusters HPDBSCAN::cluster<double>(Dataset&, int);
