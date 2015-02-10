#include "rules.h"

void merge(Rules& omp_out, Rules& omp_in)
{
    for (const auto& rule : omp_in)
    {
        omp_out.update(rule.first, rule.second);
    }
}

Rules::Rules()
{
    this->m_rules[NOISE] = 0L;
}
    
ssize_t Rules::rule(const ssize_t index) const
{
    const auto& pair = this->m_rules.find(index);
    if (pair != this->m_rules.end())
    {
        return pair->second;
    }
    return NOT_VISITED;
}

bool Rules::update(const ssize_t first, const ssize_t second)
{
    if (first <= second || first >= NOISE)
    {
        return false;
    }

    const auto& pair = this->m_rules.find(first);
    if (pair != this->m_rules.end())
    {
        if (pair->second > second)
        {
            update(pair->second, second);
            this->m_rules[first] = second;
        }
        else
        {
            update(second, pair->second );
        }
    }
    else
    {
        this->m_rules[first]  = second;
    }
    
    return true;
}
