// Round 12 four-head scorer. An epoch starts with four binary64 g values,
// then CHUNKS packed Q8 loads (head 0 occupies bits [127:0]). Epoch state
// remains immutable through the final output handshake. The caller sends
// consecutive 32-byte beats, byte 0 in the low bits, and marks the final
// physical beat. Each record contains CHUNKS*16 codes, little-endian FP16
// mean and radius scales, an unsigned count (0..32), and eleven pad bytes.
// Final high-half burst padding is ignored. Count zero marks a masked score.
`default_nettype none

module r12_frontend #(
    parameter CHUNKS = 6, COEFF_LANES = 2, PATCH = 0, PRELOADED = 0,
    parameter DIM = CHUNKS * 16
)(
    input wire clk, rst_n,
    input wire query_begin_valid,
    output wire query_begin_ready,
    input wire [255:0] query_scales,
    input wire query_valid,
    output wire query_ready,
    input wire [511:0] query_data,
    input wire input_valid, input_last,
    output wire input_ready,
    input wire [255:0] input_data,
    // Preloaded mode accepts code bytes, eight interleaved 22-bit
    // coefficients, count, then signed 32-bit bias in ascending bit order.
    input wire [8*DIM+215:0] preload_data,
    output wire output_valid,
    input wire output_ready,
    output wire [31:0] output_tag,
    output wire [7:0] output_count,
    output wire [191:0] output_scores
);
    localparam HALF_INDEX_BITS = $clog2(CHUNKS+1);
    localparam SHARED_BURST = ((CHUNKS+1) % 2) != 0;
    reg epoch_open, query_loaded, stream_closed;
    reg [2:0] query_chunk;
    reg [255:0] g_scales;
    reg [511:0] query_cache [0:CHUNKS-1];
    reg [511:0] abs_cache [0:CHUNKS-1];

    // The same two banks are held from first accepted byte through the
    // last dot launch. Each 16-byte half has one write enable and one
    // indexed read; a boundary beat can write one half into each bank.
    reg [1:0] slot_live, slot_complete, slot_prepared;
    reg write_slot, issue_slot, coefficient_slot, read_slot;
    reg [HALF_INDEX_BITS-1:0] half_index;
    reg [2:0] issue_pos, coefficient_pos, dot_chunk;
    reg [31:0] input_tag;
    reg [31:0] record_tag [0:1];
    wire [127:0] slot_code [0:1][0:CHUNKS-1];
    wire [127:0] slot_metadata [0:1];
    wire [7:0] slot_count [0:1];
    wire [31:0] slot_bias [0:1];
    wire [175:0] slot_coefficients [0:1];

    wire input_fire = input_valid && input_ready;
    wire stream_end = input_fire && input_last;
    wire end_at_low = SHARED_BURST && half_index == CHUNKS;
    wire end_at_high = half_index == CHUNKS-1;
    wire straddle = end_at_low && !input_last;
    wire complete_fire = input_fire && (PRELOADED || end_at_low || end_at_high);
    wire issue_fire;
    wire issue_last = issue_pos == 8-COEFF_LANES;
    wire coefficient_valid;
    wire coefficient_done = coefficient_valid && coefficient_pos == 8-COEFF_LANES;
    wire [21:0] returned_coeff [0:COEFF_LANES-1];
    wire [511:0] input_abs;

    // Two result credits cover both occupied FIFO entries and pages in
    // the fixed-latency tail. Once a page starts, its chunks cannot block.
    reg [1:0] outstanding, fifo_count;
    reg fifo_write, fifo_read;
    wire dot_first = dot_chunk == 0;
    wire dot_last = dot_chunk == CHUNKS-1;
    wire result_valid;
    wire [231:0] result;
    reg [231:0] result_fifo [0:1];
    wire output_fire = output_valid && output_ready;
    wire dot_fire = slot_prepared[read_slot] && (!dot_first || outstanding < 2 || output_fire);
    wire release_record = dot_fire && dot_last;

    assign query_begin_ready = rst_n && !epoch_open;
    assign query_ready = rst_n && epoch_open && !query_loaded;
    assign output_valid = rst_n && fifo_count != 0;
    assign {output_tag, output_count, output_scores} = result_fifo[fifo_read];
    assign input_ready = rst_n && query_loaded && !stream_closed &&
                         ((!PRELOADED && half_index != 0) || !slot_live[write_slot]) &&
                         (PRELOADED || !straddle || !slot_live[!write_slot]);

    genvar qlane;
    generate for (qlane=0; qlane<64; qlane=qlane+1) begin : query_magnitude
        wire [7:0] q = query_data[8*qlane +: 8];
        assign input_abs[8*qlane +: 8] = q[7] ? (~q + 8'd1) : q;
    end endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            epoch_open <= 1'b0;
            query_loaded <= 1'b0;
            stream_closed <= 1'b0;
            query_chunk <= 0;
        end else if (query_begin_valid && query_begin_ready) begin
            epoch_open <= 1'b1;
            query_loaded <= 1'b0;
            stream_closed <= 1'b0;
            query_chunk <= 0;
            g_scales <= query_scales;
        end else begin
            if (query_valid && query_ready) begin
                query_cache[query_chunk] <= query_data;
                abs_cache[query_chunk] <= input_abs;
                if (query_chunk == CHUNKS-1)
                    query_loaded <= 1'b1;
                query_chunk <= query_chunk + 3'd1;
            end
            if (stream_end)
                stream_closed <= 1'b1;
            // Final pop is the first cycle the whole epoch may retire.
            if (stream_closed && slot_live == 0 && outstanding == 1 && output_fire) begin
                epoch_open <= 1'b0;
                query_loaded <= 1'b0;
            end
        end
    end

    genvar s, h, c;
    generate
        if (PRELOADED) begin : external_coefficients
            assign issue_fire = 1'b0;
            assign coefficient_valid = 1'b0;
        end else begin : generated_coefficients
            wire [31:0] slot_scales [0:1];
            wire [COEFF_LANES-1:0] returned_valid;
            assign issue_fire = slot_complete[issue_slot];
            assign coefficient_valid = returned_valid[0];
            for (c=0; c<COEFF_LANES; c=c+1) begin : pipe
                wire [2:0] term = issue_pos + c;
                wire [63:0] g = g_scales[64*term[2:1] +: 64];
                wire [31:0] scales = slot_scales[issue_slot];
                r12_q24_coeff transform (
                    .clk(clk), .rst_n(rst_n), .in_valid(issue_fire), .g64(g),
                    .scale(term[0] ? scales[31:16] : scales[15:0]), .radius(term[0]),
                    .out_valid(returned_valid[c]), .coeff(returned_coeff[c])
                );
            end
            for (s=0; s<2; s=s+1) begin : header
                wire [127:0] metadata = slot_metadata[s];
                assign slot_scales[s] = metadata[31:0];
                assign slot_count[s] = metadata[39:32];
                r12_length_bias lookup (.count(slot_count[s]), .bias(slot_bias[s]));
            end
        end
        for (s=0; s<2; s=s+1) begin : banks
            for (h=0; h<CHUNKS+1; h=h+1) begin : pieces
                reg [127:0] data;
                if (PRELOADED) begin : prepared_input
                    if (h < CHUNKS)
                        always @(posedge clk)
                            if (input_fire && write_slot == s)
                                data <= preload_data[128*h +: 128];
                end else begin : burst_input
                    wire low_write = write_slot == s && half_index == h;
                    wire high_write = h == 0
                        ? (SHARED_BURST && straddle && write_slot != s)
                        : (write_slot == s && half_index == h-1);
                    wire [127:0] piece = low_write ? input_data[127:0] : input_data[255:128];
                    always @(posedge clk)
                        if (input_fire && (low_write || high_write))
                            data <= piece;
                end
                if (h < CHUNKS)
                    assign slot_code[s][h] = data;
                else
                    assign slot_metadata[s] = data;
            end
            if (PRELOADED) begin : external_header
                reg [215:0] values;
                always @(posedge clk)
                    if (input_fire && write_slot == s)
                        values <= preload_data[8*DIM +: 216];
                assign {slot_bias[s], slot_count[s], slot_coefficients[s]} = values;
            end else begin : generated_header
                for (h=0; h<8; h=h+1) begin : result_word
                    reg [21:0] value;
                    always @(posedge clk)
                        if (coefficient_valid && coefficient_slot == s &&
                            coefficient_pos == (h / COEFF_LANES) * COEFF_LANES)
                            value <= returned_coeff[h % COEFF_LANES];
                    assign slot_coefficients[s][22*h +: 22] = value;
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n || (query_begin_valid && query_begin_ready)) begin
            slot_live <= 0;
            slot_complete <= 0;
            slot_prepared <= 0;
            write_slot <= 0;
            issue_slot <= 0;
            coefficient_slot <= 0;
            read_slot <= 0;
            half_index <= 0;
            issue_pos <= 0;
            coefficient_pos <= 0;
            dot_chunk <= 0;
            input_tag <= 0;
        end else begin
            if (input_fire) begin
                slot_live[write_slot] <= 1'b1;
                if (!PRELOADED && straddle)
                    slot_live[!write_slot] <= 1'b1;
                if (complete_fire) begin
                    if (PRELOADED)
                        slot_prepared[write_slot] <= 1'b1;
                    else
                        slot_complete[write_slot] <= 1'b1;
                    record_tag[write_slot] <= input_tag;
                    input_tag <= input_tag + 32'd1;
                    write_slot <= !write_slot;
                    half_index <= (!PRELOADED && straddle) ? 1 : 0;
                end else begin
                    half_index <= half_index + 2;
                end
            end
            if (issue_fire) begin
                issue_pos <= issue_pos + COEFF_LANES;
                if (issue_last) begin
                    slot_complete[issue_slot] <= 1'b0;
                    issue_slot <= !issue_slot;
                end
            end
            if (coefficient_valid) begin
                coefficient_pos <= coefficient_pos + COEFF_LANES;
                if (coefficient_done) begin
                    slot_prepared[coefficient_slot] <= 1'b1;
                    coefficient_slot <= !coefficient_slot;
                end
            end
            if (dot_fire)
                dot_chunk <= dot_last ? 0 : dot_chunk + 3'd1;
            if (release_record) begin
                slot_live[read_slot] <= 1'b0;
                slot_prepared[read_slot] <= 1'b0;
                read_slot <= !read_slot;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            outstanding <= 0;
            fifo_count <= 0;
            fifo_read <= 0;
            fifo_write <= 0;
        end else begin
            case ({dot_fire && dot_first, output_fire})
                2'b10: outstanding <= outstanding + 2'd1;
                2'b01: outstanding <= outstanding - 2'd1;
                default: ;
            endcase
            case ({result_valid, output_fire})
                2'b10: fifo_count <= fifo_count + 2'd1;
                2'b01: fifo_count <= fifo_count - 2'd1;
                default: ;
            endcase
            if (result_valid) begin
                result_fifo[fifo_write] <= result;
                fifo_write <= !fifo_write;
            end
            if (output_fire)
                fifo_read <= !fifo_read;
        end
    end

    r12_gqa_core #(.PATCH(PATCH)) core (
        .clk(clk), .rst_n(rst_n),
        .in_valid(dot_fire), .in_first(dot_first), .in_last(dot_last),
        .codes(slot_code[read_slot][dot_chunk]),
        .query_lanes(query_cache[dot_chunk]), .abs_lanes(abs_cache[dot_chunk]),
        .coefficients(slot_coefficients[read_slot]),
        .bias(slot_bias[read_slot]), .tag(record_tag[read_slot]), .count(slot_count[read_slot]),
        .out_valid(result_valid), .result(result)
    );
endmodule
`default_nettype wire
