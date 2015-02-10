#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <vector>

/**
 * Atomic Operations
 */
template <typename T>
void atomicMin(T* address, T value)
{
    T previous = __sync_fetch_and_add(address, 0);
    
    while (previous > value) {
        if  (__sync_bool_compare_and_swap(address, previous, value))
        {
            break;
        }
        else
        {
            previous = __sync_fetch_and_add(address, 0);
        }
    }
}

/**
 * Vector Pointer Wrapping
 */
template <class T>
void wrap(T *source, size_t size, std::vector<T, std::allocator<T> >& target)
{
    typename std::_Vector_base<T, std::allocator<T> >::_Vector_impl* ptr = (typename std::_Vector_base<T, std::allocator<T> >::_Vector_impl*) &target;
    ptr->_M_start  = source;
    ptr->_M_finish = ptr->_M_end_of_storage = ptr->_M_start + size;
}

template <class T>
void release(std::vector<T, std::allocator<T> >& target)
{
    typename std::_Vector_base<T, std::allocator<T> >::_Vector_impl* ptr = (typename std::_Vector_base<T, std::allocator<T> >::_Vector_impl*) &target;
    ptr->_M_start = ptr->_M_finish = ptr->_M_end_of_storage = NULL;
}

/**
 * Output
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set)
{
    os << "{";
    std::copy(set.begin(), set.end(), std::ostream_iterator<T>(os, ", "));
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "[";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, ", "));
    os << "]";
    return os;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& pair)
{
    os << "[" << pair.first << " : " << pair.second << "]";
    return os;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::map<T, U>& map)
{
    for (auto& pair : map)
    {
        os << pair << std::endl;
    }
    return os;
}

#endif // UTIL_H
