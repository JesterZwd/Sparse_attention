// Round 11 event replay. The bank timings, mapping and arbiter priority are
// imported from the Round 6/7 command replay. Everything else uses femtoseconds.
#include "lpddr5_banked.cpp"

#include <vector>

namespace {
constexpr Clock ck_fs = 1250000;
constexpr int descriptor_ck = 2;
constexpr int queue_depth = 32;
constexpr int burst_bytes = 32;

int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }

enum Stage { summary, key, value, stages };
enum Mode { serial, serial_one_slot, summary_stream, double_buffer, dependency_bound };
enum Config {
    lanes, pe, mode, scorer_ii, scorer_latency, coeff_ii, coeff_latency, query_cycles,
    summary_address, summary_bursts, summary_stride, history_pages, heads, dim,
    key_tiles, value_tiles, normalize_cycles, topk_cycles, planner_cycles,
    gate_cycles, gate_scratch, context_records, pv_final_cycles, unpack_lanes, tile_capacity,
    frontend_period, dot_ii, dot_tail, whole_coeff_service, whole_coeff_latency, record_slots, beat_cycles,
    ingress_cdc, result_cdc, frontend_aux_bytes, result_slots, query_load_cycles,
    burst_fifo_slots, legacy_dot_period, legacy_burst_fifo_slots
};
enum Metric {
    start, completion, score_first, score_done, summary_read_done, normalized_topk_done,
    k_first_command, qk_first, k_read_done, qk_done, gate_done, v_first_command,
    v_first_rd, v_first_return, pv_first, v_read_done, pv_done, total_rd, total_act,
    total_pre, hits, summary_rd, k_rd, v_rd, command_cks, window_peak, descriptor_peak,
    descriptors, summary_wait, k_wait, v_wait, score_busy, qk_busy, pv_busy, k_unpack,
    v_unpack, pv_weights, dram_busy, sram_peak, sram_summary_peak, sram_k_peak,
    sram_gate_peak, sram_v_peak, summary_slots_peak, key_slots_peak, value_slots_peak,
    packed_peak, unpacked_peak, relaxed_bound, coefficient_busy, score_data_overlap, qk_data_overlap,
    pv_data_overlap, request_window_wait, summary_slot_wait, key_slot_wait, value_slot_wait,
    live_sram_peak, pv_weights_sram_peak, first_record, last_record, last_frontend_result,
    ingress_queue_wait, record_service_wait, result_cdc_wait, ingress_busy, frontend_reservation_peak,
    metric_count
};

struct Stats {
    int64_t reads = 0, acts = 0, pres = 0, hits = 0;
    Clock active = 0;
};

struct TaggedRequest {
    int64_t row, number, burst;
    int job;
};

struct TaggedBank : Bank {
    std::deque<TaggedRequest> waiting;
};

struct ReadRun {
    int64_t first, count;
    int job;
    Clock eligible;
};

struct Return {
    Clock at;
    TaggedRequest request;
};

struct Activity {
    std::array<Clock, stages> data_overlap{};
    std::array<Clock, stages> slot_wait{};
    std::deque<std::pair<Clock, Clock>> scheduled_compute;
    Clock window_wait = 0, accounted = 0, compute_start = 0, compute_end = 0;
    Stage stage = summary;
    bool slot_blocked = false;
};

struct Dram {
    std::array<TaggedBank, bank_count> banks;
    std::array<Clock, bank_group_count> group_read;
    std::array<Clock, 4> act_window;
    std::deque<ReadRun> runs;
    std::deque<Return> returns;
    std::vector<int64_t> remaining;
    std::vector<Clock> tile_ready;
    std::vector<bool> summary_received;
    std::vector<Clock> summary_return_time;
    int64_t summary_base = 0, summary_prefix = 0, summary_limit = 0, summary_returned = 0;
    int64_t phase_remaining = 0, run_offset = 0, arrivals = 0, outstanding = 0;
    Clock time = 0, command_free = 0, launch_free = 0;
    Clock first_command = -1, first_read = -1, first_return = -1, last_return = -1;
    Clock last_act = past, last_read = past, last_pre = past, active_start = 0;
    uint32_t pending_banks = 0;
    int64_t group_peak = 0, pending_peak = 0, descriptor_count = 0;
    Stats stats;
    Activity activity;

    Dram() {
        group_read.fill(past);
        act_window.fill(past);
    }

    void account(Clock until) {
        const Clock from = activity.accounted;
        activity.accounted = until;
        if (from >= until) return;
        if (activity.slot_blocked) activity.slot_wait[activity.stage] += until - from;
        if (!runs.empty()) {
            const auto& run = runs.front();
            const bool needs_credit = run.job < 0 && run_offset >= summary_limit;
            const auto eligible_time = std::max(from, run.eligible * ck_fs);
            if (needs_credit) activity.slot_wait[summary] += std::max<Clock>(0, until - eligible_time);
            else if (outstanding == queue_depth)
                activity.window_wait += std::max<Clock>(0, until - eligible_time);
        }
        if (!returns.empty()) {
            const Clock end = returns.front().at * ck_fs;
            const auto left = std::max({from, end - bl * ck_fs, activity.compute_start});
            const auto right = std::min({until, end, activity.compute_end});
            activity.data_overlap[activity.stage] += std::max<Clock>(0, right - left);
            for (const auto& span : activity.scheduled_compute) {
                const auto a = std::max({from, end - bl * ck_fs, span.first});
                const auto b = std::min({until, end, span.second});
                activity.data_overlap[activity.stage] += std::max<Clock>(0, b - a);
            }
        }
        while (!activity.scheduled_compute.empty() && activity.scheduled_compute.front().second <= until)
            activity.scheduled_compute.pop_front();
    }

    void start_phase(Stage phase, int count) {
        account(time);
        activity.stage = phase;
        activity.slot_blocked = false;
        activity.compute_start = activity.compute_end = time;
        remaining.assign(count, 0);
        tile_ready.assign(count, -1);
        first_command = first_read = first_return = last_return = -1;
    }

    void submit(int64_t first, int64_t count, int job) {
        if (job >= 0) remaining[job] += count;
        phase_remaining += count;
        launch_free = std::max(launch_free, ceil_div(time, ck_fs)) + descriptor_ck;
        runs.push_back({first, count, job, launch_free});
        ++descriptor_count;
        pending_peak = std::max<int64_t>(pending_peak, runs.size());
    }

    bool admissible() const {
        return !runs.empty() && outstanding < queue_depth &&
               (runs.front().job >= 0 || run_offset < summary_limit);
    }

    void retire() {
        while (!returns.empty() && returns.front().at * ck_fs <= time) {
            const auto req = returns.front().request;
            returns.pop_front();
            if (--outstanding == 0) stats.active += time - active_start;
            if (req.job >= 0) {
                if (--remaining[req.job] == 0) tile_ready[req.job] = time;
            } else {
                const auto index = req.burst - summary_base;
                summary_received[index] = true;
                ++summary_returned;
                if (!summary_return_time.empty()) summary_return_time[index] = time;
                while (summary_prefix < static_cast<int64_t>(summary_received.size()) &&
                       summary_received[summary_prefix]) {
                    ++summary_prefix;
                }
            }
            --phase_remaining;
            if (first_return < 0) first_return = time;
            last_return = time;
        }
    }

    void admit(Clock now) {
        while (admissible() && runs.front().eligible <= now) {
            const auto& run = runs.front();
            const auto burst = run.first + run_offset;
            const auto stripe = burst / row_bursts;
            const int id = stripe % bank_count;
            banks[id].waiting.push_back({stripe / bank_count, arrivals++, burst, run.job});
            pending_banks |= 1U << id;
            if (!outstanding) active_start = now * ck_fs;
            group_peak = std::max(group_peak, ++outstanding);
            if (++run_offset == run.count) {
                run_offset = 0;
                runs.pop_front();
            }
        }
    }

    void issue(const Candidate& candidate, Clock at) {
        if (first_command < 0) first_command = at * ck_fs;
        auto& bank = banks[candidate.bank];
        const auto req = bank.waiting.front();
        command_free = at + 1;
        switch (candidate.action) {
        case Action::read:
            if (first_read < 0) first_read = at * ck_fs;
            last_read = bank.last_read = group_read[candidate.bank % bank_group_count] = at;
            returns.push_back({at + cl + bl, req});
            stats.hits += bank.opened_for != req.number;
            bank.waiting.pop_front();
            if (bank.waiting.empty()) pending_banks &= ~(1U << candidate.bank);
            ++stats.reads;
            break;
        case Action::precharge:
            last_pre = bank.last_precharge = at;
            bank.open_row = -1;
            ++stats.pres;
            break;
        case Action::activate:
            bank.last_act = last_act = at + 1;
            act_window[stats.acts++ % 4] = last_act;
            bank.open_row = req.row;
            bank.opened_for = req.number;
            command_free = at + 2;
            break;
        }
    }

    // Stop at a compute deadline or the next data return that can unblock a consumer.
    bool advance(Clock deadline) {
        Clock now = ceil_div(time, ck_fs);
        while (true) {
            const Clock returning = returns.empty() ? never : returns.front().at;
            Clock next = returning;
            if (admissible()) next = std::min(next, std::max(now, runs.front().eligible));
            if (pending_banks) next = std::min(next, std::max(now, command_free));
            if (next == never && deadline == never) return false;
            if (next == never || next * ck_fs >= deadline) {
                account(deadline);
                time = deadline;
                retire();
                return true;
            }
            if (next == returning) {
                account(next * ck_fs);
                time = next * ck_fs;
                retire();
                return true;
            }
            now = next;
            account(now * ck_fs);
            admit(now);
            if (command_free > now) continue;

            Candidate candidate;
            Clock ready_at = never;
            for (uint32_t pending = pending_banks; pending; pending &= pending - 1) {
                const int id = __builtin_ctz(pending);
                const auto& bank = banks[id];
                const auto& req = bank.waiting.front();
                const auto [action, ready] = bank_action(
                    bank, req.row, last_read, group_read[id % bank_group_count],
                    last_pre, last_act, act_window[stats.acts % 4]);
                if (ready > now) ready_at = std::min(ready_at, ready);
                else if (candidate.prefers(action, req.number)) candidate = {id, action, req.number};
            }
            if (candidate.bank >= 0) issue(candidate, now);
            else if (pending_banks) {
                // Commands admitted by a newly ready descriptor can precede this bank deadline.
                const Clock launching = admissible() ? runs.front().eligible : never;
                const Clock data = returns.empty() ? never : returns.front().at;
                now = std::min({ready_at, launching, data});
            }
        }
    }
};

struct Tile {
    int64_t records, macs, packed, first_run, last_run;
};

struct FrontendRecord {
    Clock dot_first, reusable, result, queued;
};

struct FrontendSchedule {
    Clock period, coefficient_free = 0, dot_free = 0;
    int64_t coefficient_service, coefficient_latency, dot_service, tail, credits;
    std::deque<Clock> results;

    explicit FrontendSchedule(const int64_t* c)
        : period(c[frontend_period]), coefficient_service(c[whole_coeff_service]),
          coefficient_latency(c[whole_coeff_latency]), dot_service(c[dot_ii]),
          tail(c[dot_tail]), credits(c[result_slots]) {}

    FrontendRecord schedule(Clock complete) {
        // The complete marker registers the record before its first coefficient issue.
        const auto coefficient_start = std::max(complete + period, coefficient_free);
        coefficient_free = coefficient_start + coefficient_service * period;
        // The coefficient bank becomes readable on the edge after the final write.
        const auto prepared = coefficient_start + (coefficient_latency + 1) * period;
        auto dot = std::max(prepared, dot_free);
        if (static_cast<int64_t>(results.size()) >= credits) dot = std::max(dot, results.front());
        while (!results.empty() && results.front() <= dot) results.pop_front();
        dot_free = dot + dot_service * period;
        const auto result = dot + (dot_service - 1 + tail) * period;
        results.push_back(result);
        return {dot, dot_free, result, dot - complete - (coefficient_latency + 2) * period};
    }
};

int64_t beat_records(int64_t beats, int64_t stride, int64_t count) {
    // A nonterminal straddle needs access to both adjacent record slots.
    return std::min(count, ceil_div(beats * burst_bytes, stride));
}

struct Group {
    Dram& dram;
    const int64_t* c;
    const int64_t* ranges;
    const Tile* tiles;
    int64_t* out;
    Clock group_start;
    int64_t fixed_sram;

    Group(Dram& dram, const int64_t* config, const int64_t* ranges, const Tile* tiles,
          int64_t* out)
        : dram(dram), c(config), ranges(ranges), tiles(tiles), out(out), group_start(dram.time),
          fixed_sram(c[heads] * c[dim] * 2 + queue_depth * 16 + c[key_tiles] * 8 + 64) {
        std::fill(out, out + metric_count, 0);
        for (auto field : {score_first, k_first_command, qk_first, v_first_command,
                           v_first_rd, v_first_return, pv_first, first_record, last_record,
                           last_frontend_result}) out[field] = -1;
        out[start] = group_start;
        dram.group_peak = dram.pending_peak = dram.descriptor_count = 0;
        dram.activity = {};
        dram.activity.accounted = group_start;
    }

    Clock now() const { return dram.time - group_start; }
    Clock cycles(int64_t n) const { return n * c[pe]; }
    Clock edge(Clock moment) const { return ceil_div(moment, c[pe]) * c[pe]; }
    Clock front_cycles(int64_t n) const { return n * c[frontend_period]; }
    Clock front_edge(Clock moment) const { return ceil_div(moment, c[frontend_period]) * c[frontend_period]; }
    Clock legacy_period() const { return c[legacy_dot_period] ? c[legacy_dot_period] : c[pe]; }

    void sram(Metric stage_peak, int64_t usable, int64_t idle_reservation = 0) {
        const auto bytes = fixed_sram + usable + 16 * dram.runs.size();
        out[live_sram_peak] = std::max<int64_t>(out[live_sram_peak], bytes);
        out[stage_peak] = std::max<int64_t>(out[stage_peak], bytes + idle_reservation);
        out[sram_peak] = std::max(out[sram_peak], out[stage_peak]);
    }

    void start_summaries(bool measured) {
        dram.start_phase(summary, 0);
        dram.summary_base = c[summary_address];
        dram.summary_prefix = dram.summary_limit = dram.summary_returned = 0;
        dram.summary_received.assign(c[summary_bursts], false);
        dram.summary_return_time.assign(measured ? c[summary_bursts] : 0, 0);
        dram.submit(c[summary_address], c[summary_bursts], -1);
    }

    void finish_summaries() {
        out[summary_read_done] = dram.last_return - group_start;
        out[score_done] = now();
        // The FP32 page scores survive through normalization and heap/rank scan.
        sram(sram_summary_peak, c[history_pages] * c[heads] * 4 + c[heads] * 16);
        dram.time += cycles(c[normalize_cycles] + c[topk_cycles] + c[planner_cycles]);
        out[normalized_topk_done] = now();
    }

    bool legacy_summaries() {
        const int64_t n = c[history_pages];
        const bool barrier = c[mode] == serial;
        const int64_t capacity = barrier || c[mode] == dependency_bound ? n :
                                 c[mode] == serial_one_slot ? 1 : 2;
        const auto fifo_bytes = c[legacy_burst_fifo_slots] * burst_bytes;
        start_summaries(false);
        const Clock query_ready = edge(group_start) + cycles(c[query_cycles]);
        Clock coefficient_free = group_start, score_free = query_ready, last_result = group_start;
        std::vector<Clock> prepared;
        prepared.reserve(n);
        std::deque<Clock> held;
        int64_t accepted = 0, released = 0;
        while (accepted < n || !held.empty() || dram.phase_remaining || dram.time < last_result) {
            while (!held.empty() && held.front() <= dram.time) {
                held.pop_front();
                ++released;
            }
            const int64_t credited = std::min(n, released + capacity);
            dram.summary_limit = std::min(c[summary_bursts],
                ceil_div(credited * c[summary_stride], burst_bytes) + c[legacy_burst_fifo_slots]);
            const int64_t available = std::min({n, credited,
                                               dram.summary_prefix * burst_bytes / c[summary_stride]});
            const int64_t packed = std::min(c[summary_bursts] * burst_bytes,
                (credited - released) * c[summary_stride] + (credited > released ? burst_bytes : 0));
            const auto fifo_live = std::clamp<int64_t>(dram.summary_returned * burst_bytes -
                                                      credited * c[summary_stride], 0, fifo_bytes);
            const auto metadata = (credited - released) * (2 * 4 + 8);
            if (fifo_bytes) out[frontend_reservation_peak] = std::max(out[frontend_reservation_peak],
                                                                      packed + metadata + fifo_bytes);
            out[summary_slots_peak] = std::max(out[summary_slots_peak], credited - released);
            sram(sram_summary_peak, packed + accepted * c[heads] * 4 +
                 metadata + c[heads] * c[dim] + fifo_live, fifo_bytes - fifo_live);
            if (!barrier || !dram.phase_remaining) {
                for (auto p = static_cast<int64_t>(prepared.size()); p < available; ++p) {
                    const Clock launched = std::max(coefficient_free, edge(dram.time));
                    prepared.push_back(launched + cycles(c[coeff_latency]));
                    coefficient_free = launched + cycles(c[coeff_ii]);
                }
            }
            Clock next = held.empty() ? never : held.front();
            if (accepted == n && !dram.phase_remaining) next = std::min(next, last_result);
            if (accepted < static_cast<int64_t>(prepared.size())) {
                Clock can_start = std::max(score_free, prepared[accepted]);
                if (c[legacy_dot_period]) can_start = ceil_div(can_start, legacy_period()) * legacy_period();
                if (can_start <= dram.time) {
                    if (out[score_first] < 0) out[score_first] = now();
                    out[summary_wait] += std::max<Clock>(0, dram.time - score_free);
                    score_free = dram.time + c[scorer_ii] * legacy_period();
                    dram.activity.compute_start = dram.time;
                    dram.activity.compute_end = score_free;
                    last_result = dram.time + c[scorer_latency] * legacy_period();
                    held.push_back(c[mode] == serial_one_slot ? last_result : score_free);
                    ++accepted;
                    continue;
                }
                next = std::min(next, can_start);
            }
            if (accepted == n && held.empty() && !dram.phase_remaining && dram.time >= last_result) break;
            if (next <= dram.time) next = never;
            if (!dram.advance(next)) return false;
        }
        if (c[legacy_dot_period]) dram.time = edge(dram.time);
        out[score_busy] = n * c[scorer_ii] * legacy_period();
        out[coefficient_busy] = cycles(n * c[coeff_ii]);
        finish_summaries();
        return true;
    }

    bool measured_summaries() {
        start_summaries(true);
        const auto n = c[history_pages], stride = c[summary_stride];
        const auto capacity = std::min(n, c[mode] == serial_one_slot ? 1L : c[record_slots]);
        const auto packed_reservation = capacity * stride + burst_bytes;
        const auto reservation = packed_reservation + capacity * 32 + c[frontend_aux_bytes] +
                                 c[burst_fifo_slots] * burst_bytes;
        const auto query_ready = front_edge(edge(group_start) + cycles(c[query_cycles])) +
                                 front_cycles(c[ingress_cdc] + c[query_load_cycles]);
        Clock ingress_free = query_ready, last_result = group_start, dot_free = query_ready;
        FrontendSchedule schedule(c);
        std::deque<Clock> held;
        int64_t beats = 0, accepted = 0, released = 0;

        while (true) {
            while (!held.empty() && held.front() <= dram.time) {
                held.pop_front();
                ++released;
            }
            const auto credited = std::min(n, released + capacity);
            dram.summary_limit = std::min(c[summary_bursts], ceil_div(credited * stride, burst_bytes) + c[burst_fifo_slots]);
            const auto live_packed = std::clamp<int64_t>(dram.summary_returned * burst_bytes - released * stride,
                                                        0, packed_reservation + c[burst_fifo_slots] * burst_bytes);
            const auto live = live_packed + (credited - released) * 32 + c[frontend_aux_bytes];
            out[summary_slots_peak] = std::max(out[summary_slots_peak], credited - released);
            out[frontend_reservation_peak] = reservation;
            sram(sram_summary_peak, live + accepted * c[heads] * 4 + c[heads] * c[dim], reservation - live);

            Clock next = held.empty() ? never : held.front();
            if (beats < dram.summary_prefix && beat_records(beats + 1, stride, n) <= credited) {
                const auto visible = front_edge(dram.summary_return_time[beats]) + front_cycles(c[ingress_cdc]);
                const auto launch = std::max(ingress_free, visible);
                const auto done = launch + front_cycles(c[beat_cycles] - 1);
                if (done <= dram.time) {
                    out[ingress_queue_wait] += std::max<Clock>(0, dram.time - visible - front_cycles(c[beat_cycles] - 1));
                    ingress_free = dram.time + front_cycles(1);
                    ++beats;
                    if (accepted < std::min(n, beats * burst_bytes / stride)) {
                        const auto item = schedule.schedule(dram.time);
                        if (out[first_record] < 0) {
                            out[first_record] = now();
                            out[score_first] = item.dot_first - group_start;
                        }
                        out[last_record] = now();
                        out[record_service_wait] += item.queued;
                        out[summary_wait] += std::max<Clock>(0, item.dot_first - dot_free);
                        dot_free = item.reusable;
                        dram.activity.scheduled_compute.emplace_back(item.dot_first, dot_free);
                        last_result = item.result;
                        held.push_back(item.reusable);
                        ++accepted;
                    }
                    continue;
                }
                next = std::min(next, done);
            }
            if (accepted == n && beats == c[summary_bursts] && held.empty()) {
                const auto returned = edge(last_result) + cycles(c[result_cdc]);
                if (dram.time >= returned) break;
                next = std::min(next, returned);
            }
            if (!dram.advance(next)) return false;
        }
        out[last_frontend_result] = last_result - group_start;
        out[result_cdc_wait] = edge(last_result) + cycles(c[result_cdc]) - last_result;
        out[score_busy] = front_cycles(n * c[dot_ii]);
        out[coefficient_busy] = front_cycles(n * c[whole_coeff_service]);
        out[ingress_busy] = front_cycles(beats * c[beat_cycles]);
        finish_summaries();
        return true;
    }

    bool replay_tiles(Stage stage, int count, const Tile* input) {
        dram.start_phase(stage, count);
        if (!count) return true;
        const bool is_key = stage == key;
        const bool barrier = c[mode] == serial;
        const int capacity = barrier || c[mode] == dependency_bound ? count :
                             c[mode] == double_buffer ? 2 : 1;
        const int scratch_slots = c[mode] == dependency_bound ? count :
                                  c[mode] == double_buffer ? 2 : 1;
        const int64_t unpack_slot_bytes = 2 * c[dim] * c[tile_capacity];
        const int64_t packed_slot_bytes = c[dim] * c[tile_capacity] / 2 + burst_bytes *
            (ceil_div(c[dim] * c[tile_capacity] / 16, burst_bytes) +
             ceil_div(c[dim] * c[tile_capacity] / 32, burst_bytes));
        const Clock phase_start = dram.time;
        int issued = 0, unpacked = 0, mac_started = 0, consumed = 0;
        int64_t packed = 0, scratch = 0, logits = is_key ? 0 : c[context_records] * c[heads] * 4;
        Clock unpack_free = phase_start, mac_free = phase_start;
        std::vector<Clock> ready(count, never);
        while (consumed < count) {
            if (consumed < mac_started && mac_free <= dram.time) {
                packed -= input[consumed].packed;
                scratch -= 2 * c[dim] * input[consumed].records;
                if (is_key) logits += input[consumed].records * c[heads] * 4;
                ++consumed;
            }
            while (issued < count && issued - consumed < capacity) {
                auto& tile = input[issued];
                for (auto r = tile.first_run; r < tile.last_run; ++r)
                    dram.submit(ranges[2 * r], ranges[2 * r + 1], issued);
                packed += tile.packed;
                ++issued;
            }
            dram.activity.slot_blocked = issued < count && issued - consumed >= capacity;
            Clock next = never;
            if (unpacked < issued && unpacked - consumed < scratch_slots &&
                (!barrier || !dram.phase_remaining) && dram.tile_ready[unpacked] >= 0) {
                const auto can_unpack = std::max(unpack_free, edge(dram.time));
                if (can_unpack <= dram.time) {
                    auto& tile = input[unpacked];
                    const auto delay = cycles(ceil_div(tile.records * c[dim], c[unpack_lanes]));
                    unpack_free = ready[unpacked++] = dram.time + delay;
                    out[is_key ? k_unpack : v_unpack] += delay;
                    scratch += 2 * c[dim] * tile.records;
                } else next = std::min(next, can_unpack);
            }
            if (mac_started < unpacked) {
                const Clock can_start = std::max(mac_free, ready[mac_started]);
                if (can_start <= dram.time) {
                    auto& tile = input[mac_started++];
                    const auto mac_service = cycles(ceil_div(tile.macs, c[lanes]));
                    const auto weights = is_key ? 0 : cycles(ceil_div(tile.macs / c[dim], 4));
                    mac_free = dram.time + weights + mac_service;
                    dram.activity.compute_start = dram.time + weights;
                    dram.activity.compute_end = mac_free;
                    out[is_key ? qk_busy : pv_busy] += mac_service;
                    if (out[is_key ? qk_first : pv_first] < 0)
                        out[is_key ? qk_first : pv_first] = now() + weights;
                    out[pv_weights] += weights;
                } else next = std::min(next, can_start);
            }
            if (consumed < mac_started) next = std::min(next, mac_free);
            auto& slots = out[is_key ? key_slots_peak : value_slots_peak];
            slots = std::max<int64_t>(slots, issued - consumed);
            out[packed_peak] = std::max(out[packed_peak], packed);
            out[unpacked_peak] = std::max(out[unpacked_peak], scratch);
            const auto active_logits = is_key && mac_started > consumed
                ? input[consumed].records * c[heads] * 4 : 0;
            const auto pv_weight_bytes = !is_key && mac_started > consumed
                ? input[consumed].macs / c[dim] * 4 : 0;
            out[pv_weights_sram_peak] = std::max(out[pv_weights_sram_peak], pv_weight_bytes);
            const auto pv_scratch = is_key ? 0 : c[heads] * (c[dim] + 2) * 4 +
                                     ceil_div(c[context_records] * (c[heads] + 1), 8) + pv_weight_bytes;
            int64_t bank_reservation = 0;
            if (c[mode] != dependency_bound) {
                bank_reservation = scratch_slots * unpack_slot_bytes - scratch;
                if (!barrier) bank_reservation += capacity * packed_slot_bytes - packed;
            }
            sram(is_key ? sram_k_peak : sram_v_peak, packed + scratch + logits + active_logits + pv_scratch,
                 bank_reservation);
            if (consumed < count && !dram.advance(next)) return false;
        }
        const auto phase_work = out[is_key ? qk_busy : pv_busy] + (is_key ? 0 : out[pv_weights]);
        out[is_key ? k_wait : v_wait] = dram.time - phase_start - phase_work;
        out[is_key ? k_read_done : v_read_done] = dram.last_return - group_start;
        if (is_key) out[k_first_command] = dram.first_command - group_start;
        else {
            out[v_first_command] = dram.first_command - group_start;
            out[v_first_rd] = dram.first_read - group_start;
            out[v_first_return] = dram.first_return - group_start;
        }
        return true;
    }

    bool run() {
        const auto first = dram.stats;
        if (c[frontend_period] ? !measured_summaries() : !legacy_summaries()) return false;
        out[summary_rd] = dram.stats.reads - first.reads;
        if (!replay_tiles(key, c[key_tiles], tiles)) return false;
        out[k_rd] = dram.stats.reads - first.reads - out[summary_rd];
        out[qk_done] = now();
        sram(sram_gate_peak, c[context_records] * c[heads] * 4 + c[gate_scratch]);
        dram.time += cycles(c[gate_cycles]);
        out[gate_done] = now();
        if (!replay_tiles(value, c[value_tiles], tiles + c[key_tiles])) return false;
        out[v_rd] = dram.stats.reads - first.reads - out[summary_rd] - out[k_rd];
        if (c[value_tiles]) dram.time += cycles(c[pv_final_cycles]);
        out[pv_done] = out[completion] = now();
        dram.account(dram.time);
        out[score_data_overlap] = dram.activity.data_overlap[summary];
        out[qk_data_overlap] = dram.activity.data_overlap[key];
        out[pv_data_overlap] = dram.activity.data_overlap[value];
        out[request_window_wait] = dram.activity.window_wait;
        out[summary_slot_wait] = dram.activity.slot_wait[summary];
        out[key_slot_wait] = dram.activity.slot_wait[key];
        out[value_slot_wait] = dram.activity.slot_wait[value];

        out[total_rd] = dram.stats.reads - first.reads;
        out[total_act] = dram.stats.acts - first.acts;
        out[total_pre] = dram.stats.pres - first.pres;
        out[hits] = dram.stats.hits - first.hits;
        out[dram_busy] = dram.stats.active - first.active;
        out[command_cks] = out[total_rd] + 2 * out[total_act] + out[total_pre];
        out[window_peak] = dram.group_peak;
        out[descriptor_peak] = dram.pending_peak;
        out[descriptors] = dram.descriptor_count;

        const auto first_summary_ready = std::max(cycles(c[query_cycles]),
                                                  ceil_div(c[summary_stride], burst_bytes) * bl * ck_fs);
        const auto serve_summaries = c[frontend_period]
            ? front_cycles((c[history_pages] - 1) * std::max(c[dot_ii], c[whole_coeff_service]) +
                           c[whole_coeff_latency] + 2 + c[dot_ii] - 1 + c[dot_tail])
            : cycles(c[coeff_latency]) + (c[history_pages] - 1) *
              std::max(c[scorer_ii] * legacy_period(), cycles(c[coeff_ii])) + c[scorer_latency] * legacy_period();
        const auto ideal_summary = std::max(out[summary_rd] * bl * ck_fs, first_summary_ready + serve_summaries);
        out[relaxed_bound] = ideal_summary + cycles(c[normalize_cycles] + c[topk_cycles] +
            c[planner_cycles] + c[gate_cycles] + (c[value_tiles] ? c[pv_final_cycles] : 0)) +
            std::max({out[k_rd] * bl * ck_fs, out[qk_busy], out[k_unpack]}) +
            std::max({out[v_rd] * bl * ck_fs, out[pv_busy] + out[pv_weights], out[v_unpack]});
        return true;
    }
};
}  // namespace

extern "C" void* round11_open() { return new Dram; }
extern "C" void round11_close(void* handle) { delete static_cast<Dram*>(handle); }

extern "C" void round12_saturated_frontend(const int64_t* args, int64_t count, int64_t stride,
                                          int64_t* output) {
    int64_t c[legacy_dot_period + 1]{};
    std::copy(args, args + legacy_dot_period - frontend_period, c + frontend_period);
    FrontendSchedule schedule(c);
    const auto cycle = c[frontend_period];
    std::deque<Clock> held;
    int64_t accepted = 0, released = 0, beats = 0;
    Clock now = (c[beat_cycles] - 1) * cycle;
    while (accepted < count) {
        while (!held.empty() && held.front() <= now) {
            held.pop_front();
            ++released;
        }
        if (beat_records(beats + 1, stride, count) > released + c[record_slots]) {
            now = held.front();
            continue;
        }
        ++beats;
        if (accepted < beats * burst_bytes / stride) {
            const auto item = schedule.schedule(now);
            output[4 * accepted] = now / cycle;
            output[4 * accepted + 1] = item.dot_first / cycle;
            output[4 * accepted + 2] = item.reusable / cycle;
            output[4 * accepted + 3] = item.result / cycle;
            held.push_back(item.reusable);
            ++accepted;
        }
        now += c[beat_cycles] * cycle;
    }
}

extern "C" int replay_round11(void* handle, const int64_t* config, const int64_t* ranges,
                              const int64_t* tile_data, int64_t* output) {
    // Decode the integer interchange vector without aliasing Python's memory.
    std::vector<Tile> tiles;
    const auto count = config[key_tiles] + config[value_tiles];
    tiles.reserve(count);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t* fields = tile_data + 5 * i;
        tiles.push_back({fields[0], fields[1], fields[2], fields[3], fields[4]});
    }
    return Group(*static_cast<Dram*>(handle), config, ranges, tiles.data(), output).run() ? 0 : 1;
}
