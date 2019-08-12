/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Implementation of lock-free atomic operations
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef ATOMIC_H
#define	ATOMIC_H

#include <cstddef>
#include <functional>

template <typename T, typename O>
void _atomic_op(T* address, T value, O op) {
    T previous = __sync_fetch_and_add(address, 0);

    while (op(value, previous)) {
        if  (__sync_bool_compare_and_swap(address, previous, value)) {
            break;
        } else {
            previous = __sync_fetch_and_add(address, 0);
        }
    }
}

template <typename T>
void atomic_min(T* address, T value) {
    _atomic_op(address, value, std::less<T>());
}

template <typename T>
void atomic_max(T* address, T value) {
    _atomic_op(address, value, std::greater<T>());
}

#endif // ATOMIC_H
