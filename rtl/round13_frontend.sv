// Four-head CoMRAD scorer with an ordered burst return FIFO and independent
// record/score credits. The epoch must drain before Query or gains can change.
`default_nettype none
module r13_frontend #(
    parameter CHUNKS = 6, RECORD_SLOTS = 2, RESULT_CREDITS = 2, RX_DEPTH = 16
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
    output wire output_valid,
    input wire output_ready,
    output wire [31:0] output_tag,
    output wire [7:0] output_count,
    output wire [191:0] output_scores
);
    localparam RECORD_BITS = $clog2(RECORD_SLOTS);
    localparam RESULT_BITS = $clog2(RESULT_CREDITS);
    localparam COUNT_BITS = $clog2(RESULT_CREDITS + 1);
    localparam RX_INDEX_BITS = $clog2(RX_DEPTH);
    localparam RX_COUNT_BITS = $clog2(RX_DEPTH + 1);
    localparam HALF_INDEX_BITS = $clog2(CHUNKS + 1);
    localparam SHARED_BURST = (CHUNKS+1) % 2 != 0;

    function automatic [RECORD_BITS-1:0] next_slot(input [RECORD_BITS-1:0] index);
        next_slot = index == RECORD_SLOTS-1 ? 0 : index + 1'b1;
    endfunction
    function automatic [RESULT_BITS-1:0] next_result(input [RESULT_BITS-1:0] index);
        next_result = index == RESULT_CREDITS-1 ? 0 : index + 1'b1;
    endfunction
    function automatic [RX_INDEX_BITS-1:0] next_beat(input [RX_INDEX_BITS-1:0] index);
        next_beat = index == RX_DEPTH-1 ? 0 : index + 1'b1;
    endfunction

    reg epoch_open, query_loaded, input_closed, stream_closed;
    reg [2:0] query_chunk;
    reg [255:0] g_scales;
    reg [511:0] query_cache [0:CHUNKS-1];
    reg [511:0] abs_cache [0:CHUNKS-1];

    reg [256:0] rx_fifo [0:RX_DEPTH-1];
    reg [RX_INDEX_BITS-1:0] rx_read, rx_write;
    reg [RX_COUNT_BITS-1:0] rx_count;
    wire rx_valid = rx_count != 0;
    wire [255:0] beat_data = rx_fifo[rx_read][255:0];
    wire beat_last = rx_fifo[rx_read][256];

    // One burst may finish one record and start the next. Both then
    // occupy distinct banks, each held through its last dot launch.
    reg [RECORD_SLOTS-1:0] slot_live, slot_complete, slot_assembled, slot_prepared;
    reg [RECORD_BITS-1:0] write_slot, issue_slot, coefficient_slot, read_slot;
    wire [RECORD_BITS-1:0] following_slot = next_slot(write_slot);
    reg [HALF_INDEX_BITS-1:0] half_index;
    reg [2:0] issue_pos, coefficient_pos, dot_chunk;
    reg [31:0] input_tag;
    reg [31:0] record_tag [0:RECORD_SLOTS-1];
    wire [127:0] slot_code [0:RECORD_SLOTS-1][0:CHUNKS-1];
    wire [127:0] slot_metadata [0:RECORD_SLOTS-1];
    wire [31:0] slot_scales [0:RECORD_SLOTS-1];
    wire [7:0] slot_count [0:RECORD_SLOTS-1];
    wire signed [31:0] slot_bias [0:RECORD_SLOTS-1];
    wire [175:0] slot_coefficients [0:RECORD_SLOTS-1];

    wire end_at_low = SHARED_BURST && half_index == CHUNKS;
    wire end_at_high = half_index == CHUNKS-1;
    wire straddle = end_at_low && !beat_last;
    wire record_ready = (half_index != 0 || !slot_live[write_slot]) &&
                        (!straddle || !slot_live[following_slot]);
    wire input_fire = rx_valid && record_ready;
    wire complete_fire = input_fire && (end_at_low || end_at_high);
    wire push = input_valid && input_ready;
    wire issue_fire = slot_complete[issue_slot];
    wire issue_last = issue_pos == 6;
    wire [1:0] returned_valid;
    wire coefficient_valid = returned_valid[0];
    wire coefficient_done = coefficient_valid && coefficient_pos == 6;
    wire [21:0] returned_coeff [0:1];
    wire [511:0] input_abs;

    // Reserving a credit on the first dot makes the tail nonblocking.
    // The credit retires only when the downstream receives its score.
    reg [COUNT_BITS-1:0] outstanding, fifo_count;
    reg [RESULT_BITS-1:0] fifo_write, fifo_read;
    reg [231:0] result_fifo [0:RESULT_CREDITS-1];
    wire dot_first = dot_chunk == 0;
    wire dot_last = dot_chunk == CHUNKS-1;
    wire result_valid;
    wire [231:0] result;
    wire output_fire = output_valid && output_ready;
    wire dot_fire = slot_prepared[read_slot] &&
                    (!dot_first || outstanding < RESULT_CREDITS || output_fire);
    wire release_record = dot_fire && dot_last;
    wire begin_fire = query_begin_valid && query_begin_ready;

    assign query_begin_ready = rst_n && !epoch_open;
    assign query_ready = rst_n && epoch_open && !query_loaded;
    assign input_ready = rst_n && query_loaded && !input_closed &&
                         (rx_count < RX_DEPTH || input_fire);
    assign output_valid = rst_n && fifo_count != 0;
    assign {output_tag, output_count, output_scores} = result_fifo[fifo_read];

    always @(posedge clk) begin
        if (!rst_n) begin
            rx_read <= 0;
            rx_write <= 0;
            rx_count <= 0;
        end else begin
            case ({push, input_fire})
                2'b10: rx_count <= rx_count + 1'b1;
                2'b01: rx_count <= rx_count - 1'b1;
                default: ;
            endcase
            if (push) begin
                rx_fifo[rx_write] <= {input_last, input_data};
                rx_write <= next_beat(rx_write);
            end
            if (input_fire)
                rx_read <= next_beat(rx_read);
        end
    end

    genvar qlane;
    generate for (qlane=0; qlane<64; qlane=qlane+1) begin : query_magnitude
        wire [7:0] q = query_data[8*qlane +: 8];
        assign input_abs[8*qlane +: 8] = q[7] ? (~q + 8'd1) : q;
    end endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            epoch_open <= 1'b0;
            query_loaded <= 1'b0;
            input_closed <= 1'b0;
            stream_closed <= 1'b0;
            query_chunk <= 0;
        end else if (begin_fire) begin
            epoch_open <= 1'b1;
            query_loaded <= 1'b0;
            input_closed <= 1'b0;
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
            if (push && input_last)
                input_closed <= 1'b1;
            if (input_fire && beat_last)
                stream_closed <= 1'b1;
            if (stream_closed && slot_live == 0 && outstanding == 1 && output_fire) begin
                epoch_open <= 1'b0;
                query_loaded <= 1'b0;
            end
        end
    end

    genvar s, h, c;
    generate
        for (c=0; c<2; c=c+1) begin : coefficient_lane
            wire [2:0] term = issue_pos + c;
            wire [63:0] g = g_scales[64*term[2:1] +: 64];
            wire [31:0] scales = slot_scales[issue_slot];
            r12_q24_coeff transform (
                .clk(clk), .rst_n(rst_n), .in_valid(issue_fire), .g64(g),
                .scale(term[0] ? scales[31:16] : scales[15:0]), .radius(term[0]),
                .out_valid(returned_valid[c]), .coeff(returned_coeff[c])
            );
        end
        for (s=0; s<RECORD_SLOTS; s=s+1) begin : banks
            assign slot_scales[s] = slot_metadata[s][31:0];
            assign slot_count[s] = slot_metadata[s][39:32];
            r12_length_bias lookup (.count(slot_count[s]), .bias(slot_bias[s]));
            for (h=0; h<CHUNKS+1; h=h+1) begin : pieces
                reg [127:0] data;
                wire low_write = write_slot == s && half_index == h;
                wire high_write = h == 0
                    ? (SHARED_BURST && straddle && following_slot == s)
                    : (write_slot == s && half_index == h-1);
                always @(posedge clk)
                    if (input_fire && (low_write || high_write))
                        data <= low_write ? beat_data[127:0] : beat_data[255:128];
                if (h < CHUNKS)
                    assign slot_code[s][h] = data;
                else
                    assign slot_metadata[s] = data;
            end
            for (h=0; h<8; h=h+1) begin : result_word
                reg [21:0] value;
                always @(posedge clk)
                    if (coefficient_valid && coefficient_slot == s && coefficient_pos == (h / 2) * 2)
                        value <= returned_coeff[h % 2];
                assign slot_coefficients[s][22*h +: 22] = value;
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n || begin_fire) begin
            slot_live <= 0;
            slot_complete <= 0;
            slot_assembled <= 0;
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
                if (straddle)
                    slot_live[following_slot] <= 1'b1;
                if (complete_fire) begin
                    slot_complete[write_slot] <= 1'b1;
                    slot_assembled[write_slot] <= 1'b1;
                    record_tag[write_slot] <= input_tag;
                    input_tag <= input_tag + 32'd1;
                    write_slot <= following_slot;
                    half_index <= straddle ? 1 : 0;
                end else begin
                    half_index <= half_index + 2;
                end
            end
            if (issue_fire) begin
                issue_pos <= issue_pos + 3'd2;
                if (issue_last) begin
                    slot_complete[issue_slot] <= 1'b0;
                    issue_slot <= next_slot(issue_slot);
                end
            end
            if (coefficient_valid) begin
                coefficient_pos <= coefficient_pos + 3'd2;
                if (coefficient_done) begin
                    slot_prepared[coefficient_slot] <= 1'b1;
                    coefficient_slot <= next_slot(coefficient_slot);
                end
            end
            if (dot_fire)
                dot_chunk <= dot_last ? 0 : dot_chunk + 3'd1;
            if (release_record) begin
                slot_live[read_slot] <= 1'b0;
                slot_assembled[read_slot] <= 1'b0;
                slot_prepared[read_slot] <= 1'b0;
                read_slot <= next_slot(read_slot);
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
                2'b10: outstanding <= outstanding + 1'b1;
                2'b01: outstanding <= outstanding - 1'b1;
                default: ;
            endcase
            case ({result_valid, output_fire})
                2'b10: fifo_count <= fifo_count + 1'b1;
                2'b01: fifo_count <= fifo_count - 1'b1;
                default: ;
            endcase
            if (result_valid) begin
                result_fifo[fifo_write] <= result;
                fifo_write <= next_result(fifo_write);
            end
            if (output_fire)
                fifo_read <= next_result(fifo_read);
        end
    end

    r12_gqa_core #(.PATCH(1)) core (
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
