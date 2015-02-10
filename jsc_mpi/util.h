#ifndef UTIL_H
#define	UTIL_H

#include <stddef.h>
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

template <typename T>
void atomicMax(T* address, T value)
{
    T previous = __sync_fetch_and_add(address, 0);
    
    while (previous < value) {
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

#endif	// UTIL_H
