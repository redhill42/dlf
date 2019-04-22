#ifndef _FIBONACCI_H
#define _FIBONACCI_H

#include <unordered_map>

#define fibonacci fibonacci_i

template <typename T = int>
T fibonacci_r(int n) noexcept {
    if (n == 0)
        return 0;
    if (n == 1)
        return 1;
    if (n > 0)
        return fibonacci_r(n-1) + fibonacci_r(n-2);
    else
        return fibonacci_r(n+2) - fibonacci_r(n+1);
}

template <typename T>
T fibonacci_m(int n, std::unordered_map<int,T>& memo) noexcept {
    auto it = memo.find(n);
    if (it != memo.end()) {
        return it->second;
    }

    T res = (n > 0) ? fibonacci_m<T>(n-1, memo) + fibonacci_m<T>(n-2, memo)
                    : fibonacci_m<T>(n+2, memo) - fibonacci_m<T>(n+1, memo);
    memo[n] = res;
    return res;
}

template <typename T = int>
T fibonacci_mr(int n) {
    auto memo = std::unordered_map<int,T>{{0, T(0)}, {1, T(1)}};
    return fibonacci_m(n, memo);
}

template <typename T = int>
T fibonacci_i(const int n) noexcept {
    T a(0), b(1), c;
    int i;

    if (n >= 0) {
        for (i = 1; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        }
    } else {
        for (i = -1; i >= n; i--) {
            c = a - b;
            a = b;
            b = c;
        }
    }

    return a;
}

#endif //_FIBONACCI_H
