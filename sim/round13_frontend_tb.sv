`timescale 1ns/1ps
module tb;
    parameter CHUNKS = 6, RECORD_SLOTS = 2, RESULT_CREDITS = 2;
    localparam RX_DEPTH = 16;
    parameter NUM_EPOCHS = 2, TOTAL_PAGES = 131, TOTAL_WORDS = 459;
    localparam DIM = CHUNKS * 16;
    localparam PRELOAD_BITS = DIM * 8 + 216;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg query_begin_valid = 0, query_valid = 0;
    wire query_begin_ready, query_ready;
    reg [255:0] query_scales = 0;
    reg [511:0] query_data = 0;
    reg input_valid = 0, input_last = 0;
    wire input_ready;
    reg [255:0] input_data = 0;
    reg output_ready = 1;
    wire output_valid;
    wire [31:0] output_tag;
    wire [7:0] output_count;
    wire [191:0] output_scores;

    reg [255:0] gmem [0:NUM_EPOCHS-1];
    reg [511:0] qmem [0:NUM_EPOCHS*CHUNKS-1];
    reg [255:0] burstmem [0:TOTAL_WORDS-1];
    reg [PRELOAD_BITS-1:0] premem [0:TOTAL_PAGES-1];
    reg [199:0] expected [0:TOTAL_PAGES-1];
    integer epochs [0:NUM_EPOCHS-1];
    integer cycle = 0, received = 0, output_base = 0, epoch = 0, j, w, start_page = 0, start_word = 0;
    integer pages, words, stall_mode = 0, previous_epoch_drained = 1;
    integer active_words = 0, retired_in_epoch = 0;
    integer active_pages = 0, epochs_started = 0;
    integer dot_pages = 0, peak_slots = 0, peak_result_fifo = 0, peak_outstanding = 0;
    integer peak_rx = 0, retired_beats = 0, blocked_slot = 0, wait_coeff = 0, wait_credit = 0;
    reg held_output = 0;
    reg [231:0] previous_output;
    reg collecting_query = 0;
    reg [31:0] prng = 32'h253a71dd;
    wire [231:0] packed_output = {output_tag, output_count, output_scores};

    r13_frontend #(.CHUNKS(CHUNKS), .RECORD_SLOTS(RECORD_SLOTS),
                   .RESULT_CREDITS(RESULT_CREDITS), .RX_DEPTH(RX_DEPTH)) dut (.*);

    always @(negedge clk) begin
        if (rst_n) begin
            if ($countones(dut.slot_live) > peak_slots)
                peak_slots = $countones(dut.slot_live);
            if (int'(dut.rx_count) > peak_rx)
                peak_rx = int'(dut.rx_count);
            if (int'(dut.rx_count) > RX_DEPTH || int'(dut.outstanding) > RESULT_CREDITS ||
                int'(dut.fifo_count) > int'(dut.outstanding))
                $fatal(1, "front end occupancy exceeded declared capacity");
            if (int'(dut.fifo_count) > peak_result_fifo)
                peak_result_fifo = int'(dut.fifo_count);
            if (int'(dut.outstanding) > peak_outstanding)
                peak_outstanding = int'(dut.outstanding);
            prng <= {prng[30:0], prng[31] ^ prng[21] ^ prng[1] ^ prng[0]};
            output_ready <= !stall_mode || ((cycle % 137 >= 47) && (prng[5:3] != 3'b000) &&
                                           !(cycle >= 71 && cycle < 274));
        end
    end

    always @(posedge clk) begin
        cycle = cycle + 1;
        if (cycle > 20000)
            $fatal(1, "watchdog at cycle %0d, received %0d", cycle, received);
        if (rst_n) begin
            if (held_output && (!output_valid || previous_output !== packed_output))
                $fatal(1, "stalled output changed, cycle %0d", cycle);
            held_output = output_valid && !output_ready;
            previous_output = packed_output;
            if (!previous_epoch_drained && query_begin_ready)
                $fatal(1, "query epoch mutable with unfinished work");
            if (query_begin_ready && query_begin_valid) begin
                if (!previous_epoch_drained)
                    $fatal(1, "query changed before drain");
                output_base = received;
                active_pages = epochs[epochs_started];
                active_words = (active_pages * (DIM+16) + 31) / 32;
                retired_in_epoch = 0;
                epochs_started = epochs_started + 1;
                previous_epoch_drained = 0;
                $display("EV begin %0d", cycle);
            end
            if (query_ready && query_valid) begin
                if (!collecting_query)
                    $fatal(1, "accepted poisoned update to a live query");
                $display("EV query %0d", cycle);
            end
            if (input_valid && input_ready)
                $display("EV external_input %0d", cycle);
            if (dut.input_fire) begin
                if (retired_beats >= TOTAL_WORDS || dut.beat_data !== burstmem[retired_beats] ||
                    dut.beat_last !== (retired_in_epoch == active_words-1))
                    $fatal(1, "misordered receive FIFO data at beat %0d", retired_beats);
                retired_beats = retired_beats + 1;
                retired_in_epoch = retired_in_epoch + 1;
                $display("EV input %0d", cycle);
            end
            if (dut.rx_valid && !dut.record_ready) begin
                blocked_slot = blocked_slot + 1;
                $display("EV blocked_slot %0d", cycle);
            end
            if (dut.slot_live[dut.read_slot] && dut.dot_first &&
                !dut.slot_prepared[dut.read_slot] && dut.slot_assembled[dut.read_slot]) begin
                wait_coeff = wait_coeff + 1;
                $display("EV wait_coefficient %0d", cycle);
            end
            if (dut.slot_prepared[dut.read_slot] && dut.dot_first &&
                !dut.dot_fire) begin
                wait_credit = wait_credit + 1;
                $display("EV wait_credit %0d", cycle);
            end
            if (dut.input_fire && (dut.half_index == 0 || dut.straddle))
                $display("EV record_start %0d", cycle);
            if (input_valid && !input_ready)
                $display("EV input_block %0d", cycle);
            if (output_valid && !output_ready)
                $display("EV output_block %0d", cycle);
            if (dut.input_fire && dut.straddle)
                $display("EV carry %0d", cycle);
            if (dut.complete_fire)
                $display("EV complete %0d", cycle);
            if (dut.issue_fire)
                $display("EV issue %0d", cycle);
            if (dut.coefficient_done)
                $display("EV coefficient %0d", cycle);
            if (dut.dot_fire)
                $display("EV dot %0d", cycle);
            if (dut.dot_fire && dut.dot_first) begin
                if (dut.slot_coefficients[dut.read_slot] !== premem[dot_pages][8*DIM +: 176])
                    $fatal(1, "coefficient mismatch before first chunk of page %0d", dot_pages);
                if (dut.slot_count[dut.read_slot] !== premem[dot_pages][8*DIM+176 +: 8])
                    $fatal(1, "count mismatch before first chunk of page %0d", dot_pages);
                dot_pages = dot_pages + 1;
                $display("EV dot_first %0d", cycle);
            end
            if (dut.dot_fire && dut.dot_last)
                $display("EV dot_last %0d", cycle);
            if (output_valid && output_ready) begin
                if (output_tag !== received - output_base ||
                    {output_count, output_scores} !== expected[received])
                    $fatal(1, "result %0d tag %0d got %h expected %h",
                           received, output_tag, {output_count, output_scores}, expected[received]);
                $display("EV output %0d", cycle);
                received = received + 1;
                if (received - output_base == active_pages)
                    previous_epoch_drained = 1;
            end
        end
    end

    initial begin
        if (!$value$plusargs("stall=%d", stall_mode))
            stall_mode = 0;
        $readmemh("g64.hex", gmem);
        $readmemh("q8.hex", qmem);
        $readmemh("bursts.hex", burstmem);
        $readmemh("preload.hex", premem);
        $readmemh("scores.hex", expected);
        $readmemh("epochs.hex", epochs);
        repeat (3) @(negedge clk);
        rst_n = 1;
        for (epoch=0; epoch<NUM_EPOCHS; epoch=epoch+1) begin
            query_scales = gmem[epoch];
            query_begin_valid = 1;
            @(posedge clk);
            while (!query_begin_ready) @(posedge clk);
            @(negedge clk);
            query_begin_valid = 0;
            collecting_query = 1;
            query_valid = 0;
            for (j=0; j<CHUNKS; j=j+1) begin
                if (stall_mode && prng[7:6] == 0)
                    @(negedge clk);
                query_data = qmem[epoch*CHUNKS+j];
                query_valid = 1;
                @(posedge clk);
                while (!query_ready) @(posedge clk);
                @(negedge clk);
                query_valid = 0;
            end
            collecting_query = 0;
            query_data = ~qmem[epoch*CHUNKS];
            query_valid = 1;
            pages = epochs[epoch];
            words = (pages * (DIM + 16) + 31) / 32;
            for (w=0; w<words; w=w+1) begin
                if (stall_mode && prng[10:8] <= 2)
                    repeat (1 + int'(prng[12:11])) @(negedge clk);
                input_data = burstmem[start_word+w];
                input_last = w == words-1;
                input_valid = 1;
                @(posedge clk);
                while (!input_ready) @(posedge clk);
                @(negedge clk);
                input_valid = 0;
            end
            start_word = start_word + words;
            start_page = start_page + pages;
        end
        while (received != TOTAL_PAGES) @(negedge clk);
        repeat (4) @(negedge clk);
        if (received != TOTAL_PAGES || retired_beats != TOTAL_WORDS || !query_begin_ready)
            $fatal(1, "epoch failed to drain");
        $display("PEAK %0d %0d %0d", peak_slots, peak_result_fifo, peak_outstanding);
        $display("FIFOPEAK %0d", peak_rx);
        $display("WAITS %0d %0d %0d", blocked_slot, wait_coeff, wait_credit);
        $display("PASS %0d", received);
        $finish;
    end
endmodule
