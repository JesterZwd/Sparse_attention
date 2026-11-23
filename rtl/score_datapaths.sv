// Streaming 128-dimensional page scorers. Sixteen adjacent codes arrive
// per accepted beat, least-significant lane first. Load all eight query
// chunks before the first page, and drain in-flight scores before reloading.
// Scales are unsigned 22-bit Q24 supplied with the first beat. The signed
// Q24 bias is zero for Quest and log(valid_tokens) for Mean+Radius.
`default_nettype none

module score_query_cache #(
    parameter USE_ABS = 1
)(
    input wire clk, query_load,
    input wire [2:0] query_chunk, read_a, read_b,
    input wire [127:0] query_data,
    output wire [127:0] query_a, query_b, abs_a, abs_b
);
    reg [127:0] query_registers [0:7];
    always @(posedge clk)
        if (query_load)
            query_registers[query_chunk] <= query_data;

    assign query_a = query_registers[read_a];
    assign query_b = query_registers[read_b];

    generate if (USE_ABS) begin : magnitude
        // The unsigned magnitude of signed -128 is 128 and fits in 8 bits.
        reg [127:0] abs_registers [0:7];
        wire [127:0] loaded_abs;
        genvar lane;
        for (lane=0; lane<16; lane=lane+1) begin : query_lane
            wire [7:0] q = query_data[8*lane +: 8];
            assign loaded_abs[8*lane +: 8] = q[7] ? (~q + 8'd1) : q;
        end
        always @(posedge clk)
            if (query_load)
                abs_registers[query_chunk] <= loaded_abs;

        assign abs_a = abs_registers[read_a];
        assign abs_b = abs_registers[read_b];
    end else begin : no_magnitude
        assign abs_a = 128'b0;
        assign abs_b = 128'b0;
    end endgenerate
endmodule


module score_dot_front #(
    parameter STYLE = 0  // 0=Quest, 1=serial Mean+Radius, 2=fused Mean+Radius
)(
    input wire clk, rst_n, query_load,
    input wire [127:0] query_lanes, abs_lanes,
    input wire page_valid, page_first,
    input wire [127:0] page_codes,
    input wire [21:0] coeff_a, coeff_b,
    input wire signed [31:0] bias,
    output wire page_ready,
    output reg [2:0] chunk,
    output reg dot_valid,
    output reg signed [20:0] dot_a, dot_b,
    output reg [21:0] coeff_saved_a, coeff_saved_b,
    output reg signed [31:0] bias_saved
);
    localparam QUEST = 0;
    localparam SERIAL = 1;
    reg radius_pass;
    reg signed [20:0] partial_a, partial_b;
    // Serial reads the summary only once. The second pass consumes this
    // local two-bit-per-channel cache instead of rereading external memory.
    reg [31:0] radius_cache [0:7];
    wire [31:0] radius_write;
    wire [239:0] products_a, products_b;

    assign page_ready = rst_n && !query_load && !radius_pass;

    genvar lane;
    generate for (lane=0; lane<16; lane=lane+1) begin : channel
        wire [7:0] code = page_codes[8*lane +: 8];
        wire signed [7:0] q = query_lanes[8*lane +: 8];
        wire [7:0] magnitude = abs_lanes[8*lane +: 8];
        assign radius_write[2*lane +: 2] = code[7:6];

        if (STYLE == QUEST) begin : quest
            wire signed [3:0] selected = q[7] ? code[3:0] : code[7:4];
            wire signed [11:0] product = q * selected;
            // The low and high vectors carry independent scales. Their
            // sign-routed integer sums stay separate until the page tail.
            assign products_a[15*lane +: 15] = q[7] ? {{3{product[11]}},product} : 15'b0;
            assign products_b[15*lane +: 15] = q[7] ? 15'b0 : {{3{product[11]}},product};
        end else if (STYLE == SERIAL) begin : shared_multiplier
            wire [1:0] r = radius_cache[chunk][2*lane +: 2];
            wire signed [8:0] operand_q = radius_pass ? {1'b0,magnitude} : {q[7],q};
            wire signed [5:0] operand_k = radius_pass ? {4'b0,r} : code[5:0];
            wire signed [14:0] product = operand_q * operand_k;
            assign products_a[15*lane +: 15] = product;
            assign products_b[15*lane +: 15] = 15'b0;
        end else begin : fused
            wire signed [5:0] mean = code[5:0];
            wire signed [13:0] mean_product = q * mean;
            // R0 gates |q| and R1 gates 2|q|. An unsigned add accounts for
            // the only non-power-of-two case without a second multiplier.
            wire [9:0] low_term = code[6] ? {2'b0,magnitude} : 10'b0;
            wire [9:0] high_term = code[7] ? {1'b0,magnitude,1'b0} : 10'b0;
            wire [9:0] r_product = low_term + high_term;
            assign products_a[15*lane +: 15] = {mean_product[13],mean_product};
            assign products_b[15*lane +: 15] = {5'b0,r_product};
        end
    end endgenerate

    // Balanced tree with exactly one guard bit per level. The full-page
    // accumulator has 21 signed bits even for K6=-32 and Q8=-128.
    function automatic signed [18:0] reduce_lanes(input [239:0] products);
        reg signed [15:0] pair_sum [0:7];
        reg signed [16:0] quad_sum [0:3];
        reg signed [17:0] half_sum [0:1];
        integer i;
        begin
            for (i=0; i<8; i=i+1)
                pair_sum[i] = $signed(products[30*i +: 15]) +
                              $signed(products[30*i+15 +: 15]);
            for (i=0; i<4; i=i+1)
                quad_sum[i] = pair_sum[2*i] + pair_sum[2*i+1];
            for (i=0; i<2; i=i+1)
                half_sum[i] = quad_sum[2*i] + quad_sum[2*i+1];
            reduce_lanes = half_sum[0] + half_sum[1];
        end
    endfunction

    wire signed [20:0] tree_a = reduce_lanes(products_a);
    wire signed [20:0] tree_b = reduce_lanes(products_b);
    wire signed [20:0] next_a = (chunk==0 ? 21'sb0 : partial_a) + tree_a;
    wire signed [20:0] next_b = (chunk==0 ? 21'sb0 : partial_b) +
                               (STYLE==SERIAL ? tree_a : tree_b);
    wire accept = page_valid && page_ready;

    always @(posedge clk) begin
        dot_valid <= 1'b0;
        if (!rst_n) begin
            chunk <= 3'd0;
            radius_pass <= 1'b0;
        end else if (radius_pass) begin
            if (chunk == 3'd7) begin
                chunk <= 3'd0;
                radius_pass <= 1'b0;
                dot_a <= partial_a;
                dot_b <= next_b;
                dot_valid <= 1'b1;
            end else begin
                partial_b <= next_b;
                chunk <= chunk + 3'd1;
            end
        end else if (accept) begin
            if (page_first) begin
                coeff_saved_a <= coeff_a;
                coeff_saved_b <= coeff_b;
                bias_saved <= bias;
            end
            partial_a <= next_a;
            if (STYLE == SERIAL)
                radius_cache[chunk] <= radius_write;
            else
                partial_b <= next_b;
            if (chunk == 3'd7) begin
                chunk <= 3'd0;
                if (STYLE == SERIAL) begin
                    radius_pass <= 1'b1;
                end else begin
                    dot_a <= next_a;
                    dot_b <= next_b;
                    dot_valid <= 1'b1;
                end
            end else begin
                chunk <= chunk + 3'd1;
            end
        end
    end
endmodule


module score_scale_tail(
    input wire clk, rst_n, dot_valid,
    input wire signed [20:0] dot_a, dot_b,
    input wire [21:0] coeff_saved_a, coeff_saved_b,
    input wire signed [31:0] bias_saved,
    output reg score_valid,
    output reg signed [47:0] score
);
    reg scale_valid;
    reg signed [43:0] scaled_a, scaled_b;
    reg signed [31:0] scaled_bias;

    always @(posedge clk) begin
        if (!rst_n) begin
            scale_valid <= 1'b0;
            score_valid <= 1'b0;
        end else begin
            scale_valid <= dot_valid;
            score_valid <= scale_valid;
            if (dot_valid) begin
                scaled_a <= dot_a * $signed({1'b0,coeff_saved_a});
                scaled_b <= dot_b * $signed({1'b0,coeff_saved_b});
                scaled_bias <= bias_saved;
            end
            if (scale_valid)
                score <= $signed(scaled_a) + $signed(scaled_b) + $signed(scaled_bias);
        end
    end
endmodule


`define SCORE_PORTS \
    input wire clk, rst_n, query_load, \
    input wire [2:0] query_chunk, \
    input wire [127:0] query_data, \
    input wire page_valid, page_first, \
    input wire [127:0] page_codes, \
    input wire [21:0] coeff_a, coeff_b, \
    input wire signed [31:0] bias, \
    output wire page_ready, score_valid, \
    output wire signed [47:0] score

module score_stream #(parameter STYLE=0)(`SCORE_PORTS);
    wire [2:0] chunk;
    wire [127:0] q, mag;
    wire dot_valid;
    wire signed [20:0] dot_a,dot_b;
    wire [21:0] coeff_saved_a,coeff_saved_b;
    wire signed [31:0] bias_saved;
    score_query_cache #(.USE_ABS(STYLE!=0)) cache (
        .clk(clk), .query_load(query_load), .query_chunk(query_chunk),
        .query_data(query_data), .read_a(chunk), .read_b(3'd0),
        .query_a(q), .query_b(), .abs_a(mag), .abs_b()
    );
    score_dot_front #(.STYLE(STYLE)) front (
        .query_lanes(q), .abs_lanes(mag), .*
    );
    score_scale_tail tail (.*);
endmodule

module score_quest(`SCORE_PORTS);
    score_stream #(.STYLE(0)) impl (.*);
endmodule

module score_mr_serial(`SCORE_PORTS);
    score_stream #(.STYLE(1)) impl (.*);
endmodule

module score_mr_fused(`SCORE_PORTS);
    score_stream #(.STYLE(2)) impl (.*);
endmodule

module score_mr_serial_equal(`SCORE_PORTS);
    // At full input rate one front accumulates the mean of the next page
    // while the other consumes cached radii from the previous page. Query
    // and scaling resources are shared. Fixed eight-cycle radius passes
    // keep both streams' completion order without an output reorder buffer.
    reg turn;
    reg [2:0] accepted_chunk;
    wire [1:0] ready, done;
    wire [2:0] read_chunk [0:1];
    wire [127:0] qs [0:1];
    wire [127:0] mags [0:1];
    wire signed [20:0] sums_a [0:1];
    wire signed [20:0] sums_b [0:1];
    wire [21:0] scales_a [0:1];
    wire [21:0] scales_b [0:1];
    wire signed [31:0] biases [0:1];

    assign page_ready = ready[turn];
    always @(posedge clk) begin
        if (!rst_n) begin
            turn <= 1'b0;
            accepted_chunk <= 3'd0;
        end else if (page_valid && page_ready) begin
            if (accepted_chunk == 3'd7)
                turn <= ~turn;
            accepted_chunk <= accepted_chunk + 3'd1;
        end
    end

    score_query_cache cache (
        .clk(clk), .query_load(query_load), .query_chunk(query_chunk), .query_data(query_data),
        .read_a(read_chunk[0]), .read_b(read_chunk[1]),
        .query_a(qs[0]), .query_b(qs[1]), .abs_a(mags[0]), .abs_b(mags[1])
    );
    genvar f;
    generate for (f=0; f<2; f=f+1) begin : dual_front
        score_dot_front #(.STYLE(1)) front (
            .query_lanes(qs[f]), .abs_lanes(mags[f]),
            .page_valid(page_valid && turn==f), .page_ready(ready[f]),
            .chunk(read_chunk[f]), .dot_valid(done[f]), .dot_a(sums_a[f]), .dot_b(sums_b[f]),
            .coeff_saved_a(scales_a[f]), .coeff_saved_b(scales_b[f]), .bias_saved(biases[f]), .*
        );
    end endgenerate

    score_scale_tail tail (
        .dot_valid(|done), .dot_a(sums_a[done[1]]), .dot_b(sums_b[done[1]]),
        .coeff_saved_a(scales_a[done[1]]), .coeff_saved_b(scales_b[done[1]]),
        .bias_saved(biases[done[1]]), .*
    );
endmodule

`undef SCORE_PORTS
`default_nettype wire
