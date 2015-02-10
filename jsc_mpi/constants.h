#ifndef CONSTANTS_H
#define	CONSTANTS_H

#include <array>
#include <limits>
#include <map>
#include <vector>

typedef long                                       ssize_t;
typedef std::array<size_t, 4>                      CellBounds;
typedef std::array<size_t, 2>                      ComputeBounds;
typedef size_t                                     Cell;
typedef ssize_t                                    Cluster;
typedef float                                      Coord;
typedef std::map<Cell, size_t>                     CellCounter;
typedef std::map<Cell, std::pair<size_t, size_t> > CellIndex;
typedef std::vector<std::pair<size_t,size_t> >     Cuts;

static const ssize_t NOT_VISITED = std::numeric_limits<ssize_t>::max();
static const ssize_t NOISE       = std::numeric_limits<ssize_t>::max() - 1;

#endif	/* CONSTANTS_H */

