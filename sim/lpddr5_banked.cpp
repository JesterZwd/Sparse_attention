// Single-channel, read-only LPDDR5-6400 BG replay with the Round 6 mapping.
// The timings are in CK and use the pinned Ramulator preset documented there.
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>

namespace {
using Clock = int64_t;

constexpr int bank_count = 16;
constexpr int bank_group_count = 4;
constexpr int row_bursts = 64;
constexpr Clock never = std::numeric_limits<Clock>::max() / 4;
constexpr Clock past = -never;
constexpr Clock rcd = 15, rp = 15, ras = 34, rc = 49, rtp = 6, ppd = 2;
constexpr Clock cl = 17, bl = 2, ccd_s = 2, ccd_l = 4, rrd = 4, faw = 16;

enum class Action { read, precharge, activate };

struct Request {
    int64_t row;
    int64_t number;
};

struct Bank {
    std::deque<Request> fifo;
    int64_t open_row = -1;
    int64_t opened_for = -1;
    Clock last_act = past;
    Clock last_read = past;
    Clock last_precharge = past;
};

struct ScheduledAction {
    Action action;
    Clock ready;
};

ScheduledAction bank_action(const Bank& bank, int64_t row, Clock last_read,
                            Clock group_read, Clock last_precharge, Clock last_act,
                            Clock fourth_last_act) {
    if (bank.open_row == row) {
        return {Action::read, std::max({bank.last_act + rcd, last_read + ccd_s, group_read + ccd_l})};
    }
    if (bank.open_row >= 0) {
        return {Action::precharge, std::max({bank.last_act + ras, bank.last_read + rtp,
                                             last_precharge + ppd})};
    }
    // ACT2 carries the timing constraints; ACT1 takes the preceding command slot.
    return {Action::activate, std::max({bank.last_precharge + rp, bank.last_act + rc,
                                        last_act + rrd, fourth_last_act + faw}) - 1};
}

struct Candidate {
    int bank = -1;
    Action action = Action::activate;
    int64_t request = std::numeric_limits<int64_t>::max();

    bool prefers(Action next, int64_t number) const {
        if (bank < 0) return true;
        if ((action == Action::read) != (next == Action::read)) {
            return next == Action::read;
        }
        return number < request;
    }
};
}  // namespace

extern "C" void replay_banked(const int64_t* starts, const int64_t* counts,
                              const int64_t* bounds, const int64_t* waits,
                              int64_t phase_count, int64_t descriptor,
                              int64_t queue_depth, int64_t* output) {
    std::array<Bank, bank_count> banks;
    std::array<Clock, bank_group_count> group_read;
    std::array<Clock, 4> act_window;
    group_read.fill(past);
    act_window.fill(past);
    // RD return latencies are equal, so the return queue stays ordered.
    std::deque<Clock> returns;
    Clock last_act = past, last_read = past, last_precharge = past, min_precharge_gap = never, now = 0;
    int64_t outstanding = 0, peak_outstanding = 0, arrivals = 0;
    int64_t bursts = 0, hits = 0, activations = 0, precharges = 0;
    uint32_t active_banks = 0;

    for (int64_t phase = 0; phase < phase_count; ++phase) {
        now += waits[phase];
        output[phase] = now;
        Clock first_command = -1, first_read = -1;
        Clock descriptor_ready = now + descriptor;
        auto run = bounds[phase];
        int64_t offset = 0;

        while (run < bounds[phase + 1] || outstanding) {
            while (!returns.empty() && returns.front() <= now) {
                returns.pop_front();
                --outstanding;
            }
            while (run < bounds[phase + 1] && descriptor_ready <= now && outstanding < queue_depth) {
                const auto stripe = (starts[run] + offset) / row_bursts;
                const int bank = stripe % bank_count;
                banks[bank].fifo.push_back({stripe / bank_count, arrivals++});
                active_banks |= 1U << bank;
                peak_outstanding = std::max(peak_outstanding, ++outstanding);
                if (++offset == counts[run]) {
                    offset = 0;
                    ++run;
                    descriptor_ready += descriptor;
                }
            }
            if (!outstanding && run == bounds[phase + 1]) break;

            Clock next = returns.empty() ? never : returns.front();
            if (run < bounds[phase + 1] && outstanding < queue_depth) {
                next = std::min(next, descriptor_ready);
            }
            Candidate chosen;
            for (uint32_t pending = active_banks; pending; pending &= pending - 1) {
                const int id = __builtin_ctz(pending);
                const auto& bank = banks[id];
                const auto& request = bank.fifo.front();
                const auto [action, ready] = bank_action(
                    bank, request.row, last_read, group_read[id % bank_group_count],
                    last_precharge, last_act, act_window[activations % 4]);
                if (ready > now) {
                    next = std::min(next, ready);
                } else if (chosen.prefers(action, request.number)) {
                    chosen = {id, action, request.number};
                }
            }

            if (chosen.bank < 0) {
                now = next;
                continue;
            }
            if (first_command < 0) first_command = now;
            auto& bank = banks[chosen.bank];
            if (chosen.action == Action::read) {
                if (first_read < 0) first_read = now;
                last_read = bank.last_read = group_read[chosen.bank % bank_group_count] = now;
                returns.push_back(now + cl + bl);
                hits += bank.opened_for != bank.fifo.front().number;
                bank.fifo.pop_front();
                if (bank.fifo.empty()) active_banks &= ~(1U << chosen.bank);
                ++bursts;
                ++now;
            } else if (chosen.action == Action::precharge) {
                if (precharges) min_precharge_gap = std::min(min_precharge_gap, now - last_precharge);
                last_precharge = now;
                bank.last_precharge = now;
                bank.open_row = -1;
                ++precharges;
                ++now;
            } else {
                bank.last_act = last_act = now + 1;
                act_window[activations % 4] = last_act;
                bank.open_row = bank.fifo.front().row;
                bank.opened_for = bank.fifo.front().number;
                ++activations;
                now += 2;
            }
        }
        output[phase_count + phase] = first_command;
        output[2 * phase_count + phase] = first_read;
        output[3 * phase_count + phase] = now;
    }

    output += 4 * phase_count;
    output[0] = bursts;
    output[1] = hits;
    output[2] = activations;
    output[3] = precharges;
    output[4] = peak_outstanding;
    output[5] = bursts + 2 * activations + precharges;
    output[6] = precharges > 1 ? min_precharge_gap : -1;
}
