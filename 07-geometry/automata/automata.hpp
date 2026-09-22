#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace samples::automata
{
struct BinaryDfa
{
    std::vector<std::array<std::size_t, 2>> next;
    std::vector<bool> accepting;
    std::size_t start = 0;
    void validate() const;
    bool accepts(std::string_view bits) const;
};
BinaryDfa divisible_by(std::size_t divisor, bool allow_empty = false);
// ECMAScript-compatible expression. Use std::regex_match for whole-string matching.
// State elimination can grow exponentially, so construction has explicit budgets.
std::string to_regex(const BinaryDfa &dfa, std::size_t max_label_bytes = 250000,
                     std::size_t max_total_bytes = 8000000);
} // namespace samples::automata
