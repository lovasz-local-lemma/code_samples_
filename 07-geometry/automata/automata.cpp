#include "samples/automata.hpp"
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

namespace samples::automata
{
void BinaryDfa::validate() const
{
    if (next.empty() || accepting.size() != next.size() || start >= next.size())
        throw std::invalid_argument("Invalid DFA state arrays or start state");
    for (const auto &edges : next)
        for (auto destination : edges)
            if (destination >= next.size())
                throw std::invalid_argument("Invalid DFA transition");
}
bool BinaryDfa::accepts(std::string_view bits) const
{
    validate();
    auto state = start;
    for (auto c : bits)
    {
        if (c != '0' && c != '1')
            return false;
        state = next[state][std::size_t(c - '0')];
    }
    return accepting[state];
}
BinaryDfa divisible_by(std::size_t divisor, bool allow_empty)
{
    if (!divisor || divisor > 128)
        throw std::invalid_argument("This sample supports divisors 1..128");
    BinaryDfa dfa;
    dfa.next.resize(divisor + (allow_empty ? 0 : 1));
    dfa.accepting.resize(dfa.next.size(), false);
    dfa.accepting[0] = true;
    for (std::size_t r = 0; r < divisor; ++r)
        dfa.next[r] = {(2 * r) % divisor, (2 * r + 1) % divisor};
    if (!allow_empty)
    {
        dfa.start = divisor; // Nonaccepting entry state consumes the first digit.
        dfa.next.back() = dfa.next[0];
    }
    return dfa;
}

namespace
{
// nullopt denotes the empty language; "" denotes epsilon. They are not interchangeable.
using Label = std::optional<std::string>;
struct RegexAlgebra
{
    std::size_t limit;
    Label checked(std::string value) const
    {
        if (value.size() > limit)
            throw std::length_error("Regex label budget exceeded");
        return value;
    }
    Label either(const Label &a, const Label &b) const
    {
        if (!a)
            return b;
        if (!b || a == b)
            return a;
        return checked("(?:" + *a + "|" + *b + ")");
    }
    Label sequence(const Label &a, const Label &b) const
    {
        if (!a || !b)
            return std::nullopt;
        return checked(*a + *b);
    }
    Label repeat(const Label &a) const
    {
        if (!a || a->empty())
            return std::string{};
        return checked("(?:" + *a + ")*");
    }
};
} // namespace

std::string to_regex(const BinaryDfa &dfa, std::size_t max_label, std::size_t max_total)
{
    dfa.validate();
    if (dfa.next.size() > 130 || !max_label || !max_total)
        throw std::invalid_argument("Invalid regex construction budget or too many states");
    const auto count = dfa.next.size() + 2;
    const auto entry = count - 2, exit = count - 1;
    std::vector<std::vector<Label>> edges(count, std::vector<Label>(count));
    std::vector<bool> active(count, true);
    const RegexAlgebra re{max_label};
    std::size_t bytes = 0;
    const auto assign = [&](Label &target, Label value)
    {
        const auto old_size = target ? target->size() : 0;
        const auto new_size = value ? value->size() : 0;
        const auto retained = bytes - old_size;
        if (new_size > max_total - retained)
            throw std::length_error("Regex total budget exceeded");
        target = std::move(value);
        bytes = retained + new_size;
    };
    for (std::size_t i = 0; i < dfa.next.size(); ++i)
    {
        for (unsigned bit = 0; bit < 2; ++bit)
        {
            auto &edge = edges[i][dfa.next[i][bit]];
            assign(edge, re.either(edge, std::string(1, char('0' + bit))));
        }
        if (dfa.accepting[i])
            edges[i][exit] = std::string{};
    }
    edges[entry][dfa.start] = std::string{};
    for (std::size_t remaining = dfa.next.size(); remaining; --remaining)
    {
        std::size_t remove = 0, best = std::numeric_limits<std::size_t>::max();
        for (std::size_t k = 0; k < dfa.next.size(); ++k)
        {
            if (!active[k])
                continue;
            std::size_t incoming = 0, outgoing = 0;
            for (std::size_t i = 0; i < count; ++i)
            {
                if (!active[i] || i == k)
                    continue;
                incoming += edges[i][k].has_value();
                outgoing += edges[k][i].has_value();
            }
            if (incoming * outgoing < best)
            {
                best = incoming * outgoing;
                remove = k;
            }
        }
        const auto loop = re.repeat(edges[remove][remove]);
        // R_ij <- R_ij | R_ik (R_kk)* R_kj. Exclude k from both update axes.
        for (std::size_t i = 0; i < count; ++i)
        {
            if (!active[i] || i == remove || !edges[i][remove])
                continue;
            const auto prefix = re.sequence(edges[i][remove], loop);
            for (std::size_t j = 0; j < count; ++j)
            {
                if (!active[j] || j == remove || !edges[remove][j])
                    continue;
                assign(edges[i][j], re.either(edges[i][j], re.sequence(prefix, edges[remove][j])));
            }
        }
        active[remove] = false;
        for (std::size_t i = 0; i < count; ++i)
        {
            assign(edges[i][remove], std::nullopt);
            assign(edges[remove][i], std::nullopt);
        }
    }
    return edges[entry][exit].value_or("(?!)"); // Empty language: always-failing lookahead.
}
} // namespace samples::automata
