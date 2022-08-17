#include "common.hpp"

#include <numeric>

// Python-like range functions
std::vector<int> range(const int& start, const int& end) {
    std::vector<int> out(end - start);
    std::iota(out.begin(), out.end(), start);
    return out;
}

std::vector<int> range(const int& end) {
    return range(0, end);
}