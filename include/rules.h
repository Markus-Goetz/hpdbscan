/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Cluster remapping rules
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef RULES_H
#define RULES_H

#include <omp.h>
#include <unordered_map>

#include "constants.h"

class Rules {
    std::unordered_map<Cluster, Cluster> m_rules;

public:
    Rules() {
        m_rules[NOISE] = 0;
    }

    inline const std::unordered_map<Cluster, Cluster>::const_iterator begin() const {
        return m_rules.begin();
    }

    inline const std::unordered_map<Cluster, Cluster>::const_iterator end() const {
        return m_rules.end();
    }

    inline void remove(const Cluster index) {
        m_rules.erase(m_rules.find(index));
    }

    Cluster rule(const Cluster cluster) const {
        const auto& pair = m_rules.find(cluster);
        if (pair != m_rules.end()) {
            return pair->second;
        }
        return NOT_VISITED;
    }

    inline size_t size() const {
        return m_rules.size();
    }

    bool update(const Cluster first, const Cluster second) {
        if (first <= second or first >= NOISE) {
            return false;
        }
        const auto& pair = m_rules.find(first);
        if (pair != m_rules.end()) {
            if (pair->second > second) {
                update(pair->second, second);
                m_rules[first] = second;
            } else {
                update(second, pair->second );
            }
        } else {
            m_rules[first]  = second;
        }

        return true;
    }
};

void merge(Rules& omp_out, Rules& omp_in) {
    for (const auto& rule : omp_in) {
        omp_out.update(rule.first, rule.second);
    }
}
#pragma omp declare reduction(merge: Rules: merge(omp_out, omp_in)) initializer(omp_priv(omp_orig))

#endif // RULES_H
