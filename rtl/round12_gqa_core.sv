// 16 channel lanes per Q head, four heads per KV group. The same code
// beat broadcasts to all heads. Means use signed 8x6 products, radii
// use unsigned bit-gated |Q| and 2|Q|. No approximation is made in the
// integer accumulators or signed 48-bit Q24 scores.
`default_nettype none

module r12_gqa_core #(parameter PATCH=0)(
    input wire clk, rst_n, in_valid, in_first, in_last,
    input wire [127:0] codes,
    input wire [511:0] query_lanes, abs_lanes,
    input wire [175:0] coefficients,
    input wire signed [31:0] bias,
    input wire [31:0] tag,
    input wire [7:0] count,
    output reg out_valid,
    output reg [231:0] result
);
    reg [127:0] read_codes;
    reg [511:0] read_q, read_abs;
    // Coefficients and page ID are used only on the last chunk. Pipeline
    // them with that chunk so the input record can be released immediately.
    reg [247:0] read_meta, product_meta, quad_meta, tree_meta, sum_meta;
    reg read_valid, read_first, read_last;
    reg product_valid, product_first, product_last;
    reg quad_valid, quad_first, quad_last;
    reg tree_valid, tree_first, tree_last;
    reg sum_valid, partial_valid, scale_valid, compress_valid;
    reg [71:0] partial_meta, scale_meta;
    reg [39:0] compress_meta;
    wire [191:0] scores;

    always @(posedge clk) begin
        if (!rst_n) begin
            read_valid <= 1'b0;
            product_valid <= 1'b0;
            quad_valid <= 1'b0;
            tree_valid <= 1'b0;
            sum_valid <= 1'b0;
            partial_valid <= 1'b0;
            scale_valid <= 1'b0;
            compress_valid <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            read_valid <= in_valid;
            product_valid <= read_valid;
            quad_valid <= product_valid;
            tree_valid <= quad_valid;
            sum_valid <= tree_valid && tree_last;
            partial_valid <= sum_valid;
            scale_valid <= PATCH ? partial_valid : sum_valid;
            compress_valid <= scale_valid;
            out_valid <= PATCH ? compress_valid : scale_valid;
        end
        if (in_valid) begin
            read_codes <= codes;
            read_q <= query_lanes;
            read_abs <= abs_lanes;
            read_first <= in_first;
            read_last <= in_last;
            if (in_last)
                read_meta <= {tag, count, bias, coefficients};
        end
        if (read_valid) begin
            product_first <= read_first;
            product_last <= read_last;
            if (read_last)
                product_meta <= read_meta;
        end
        if (product_valid) begin
            quad_first <= product_first;
            quad_last <= product_last;
            if (product_last)
                quad_meta <= product_meta;
        end
        if (quad_valid) begin
            tree_first <= quad_first;
            tree_last <= quad_last;
            if (quad_last)
                tree_meta <= quad_meta;
        end
        if (tree_valid && tree_last)
            sum_meta <= tree_meta;
        if (sum_valid)
            partial_meta <= sum_meta[247:176];
        if (PATCH ? partial_valid : sum_valid)
            scale_meta <= PATCH ? partial_meta : sum_meta[247:176];
        if (scale_valid)
            compress_meta <= scale_meta[71:32];
        if (PATCH ? compress_valid : scale_valid)
            result <= {PATCH ? compress_meta : scale_meta[71:32], scores};
    end

    genvar h, l, group;
    generate for (h=0; h<4; h=h+1) begin : head
        wire signed [13:0] mean_product [0:15];
        wire [8:0] radius_product [0:15];
        reg signed [13:0] product_a [0:15];
        reg [8:0] product_b [0:15];
        reg signed [15:0] quad_a [0:3];
        reg [10:0] quad_b [0:3];
        reg signed [17:0] tree_a;
        reg [12:0] tree_b;
        reg signed [20:0] accumulated_a, accumulated_b;
        reg signed [20:0] sum_a, sum_b;
        reg signed [43:0] scaled_a, scaled_b;

        for (l=0; l<16; l=l+1) begin : lane
            wire signed [7:0] q = read_q[128*h+8*l +: 8];
            wire [7:0] magnitude = read_abs[128*h+8*l +: 8];
            wire [7:0] code = read_codes[8*l +: 8];
            wire signed [5:0] mean = code[5:0];
            wire [8:0] low_term = code[6] ? {1'b0,magnitude} : 9'b0;
            wire [8:0] high_term = code[7] ? {magnitude,1'b0} : 9'b0;
            assign mean_product[l] = q * mean;
            assign radius_product[l] = low_term + high_term;
            always @(posedge clk)
                if (read_valid) begin
                    product_a[l] <= mean_product[l];
                    product_b[l] <= radius_product[l];
                end
        end
        for (group=0; group<4; group=group+1) begin : quad
            wire signed [14:0] pair_a0 = product_a[group*4] + product_a[group*4+1];
            wire signed [14:0] pair_a1 = product_a[group*4+2] + product_a[group*4+3];
            wire [9:0] pair_b0 = product_b[group*4] + product_b[group*4+1];
            wire [9:0] pair_b1 = product_b[group*4+2] + product_b[group*4+3];
            always @(posedge clk)
                if (product_valid) begin
                    quad_a[group] <= pair_a0 + pair_a1;
                    quad_b[group] <= pair_b0 + pair_b1;
                end
        end

        wire signed [16:0] half_a0 = quad_a[0] + quad_a[1];
        wire signed [16:0] half_a1 = quad_a[2] + quad_a[3];
        wire [11:0] half_b0 = quad_b[0] + quad_b[1];
        wire [11:0] half_b1 = quad_b[2] + quad_b[3];
        wire signed [20:0] next_a = (tree_first ? 21'sd0 : accumulated_a) + tree_a;
        wire signed [20:0] next_b = (tree_first ? 21'sd0 : accumulated_b) + $signed({1'b0,tree_b});
        wire signed [22:0] ca = $signed({1'b0,sum_meta[44*h +: 22]});
        wire signed [22:0] cb = $signed({1'b0,sum_meta[44*h+22 +: 22]});

        always @(posedge clk) begin
            if (quad_valid) begin
                tree_a <= half_a0 + half_a1;
                tree_b <= half_b0 + half_b1;
            end
            if (tree_valid) begin
                accumulated_a <= next_a;
                accumulated_b <= next_b;
                if (tree_last) begin
                    sum_a <= next_a;
                    sum_b <= next_b;
                end
            end
        end
        if (!PATCH) begin : original_tail
            always @(posedge clk)
                if (sum_valid) begin
                    scaled_a <= sum_a * ca;
                    scaled_b <= sum_b * cb;
                end
            assign scores[48*h +: 48] = $signed(scaled_a) + $signed(scaled_b) +
                                          $signed(scale_meta[31:0]);
        end else begin : pipelined_tail
            // The original OpenSTA path ended at scaled_b. Split the
            // exact unsigned 22-bit coefficient into two 11-bit operands.
            // Each partial is signed because the accumulated dot can be
            // negative. The added recombination stage preserves all bits.
            reg signed [32:0] partial_product_a_low, partial_product_a_high;
            reg signed [32:0] partial_product_b_low, partial_product_b_high;
            wire signed [43:0] high_a = {partial_product_a_high, 11'b0};
            wire signed [43:0] high_b = {partial_product_b_high, 11'b0};
            wire signed [47:0] extend_a = scaled_a;
            wire signed [47:0] extend_b = scaled_b;
            wire signed [47:0] extend_bias = $signed(scale_meta[31:0]);
            reg [47:0] compress_sum, compress_carry;
            always @(posedge clk) begin
                if (sum_valid) begin
                    partial_product_a_low <= sum_a * $signed({1'b0,ca[10:0]});
                    partial_product_a_high <= sum_a * $signed({1'b0,ca[21:11]});
                    partial_product_b_low <= sum_b * $signed({1'b0,cb[10:0]});
                    partial_product_b_high <= sum_b * $signed({1'b0,cb[21:11]});
                end
                if (partial_valid) begin
                    scaled_a <= high_a + partial_product_a_low;
                    scaled_b <= high_b + partial_product_b_low;
                end
                // The runner-up endpoint was the chained score addition.
                // Compress the three operands before the final carry pass.
                if (scale_valid) begin
                    compress_sum <= extend_a ^ extend_b ^ extend_bias;
                    compress_carry <= ((extend_a & extend_b) | (extend_a & extend_bias) |
                                       (extend_b & extend_bias)) << 1;
                end
            end
            assign scores[48*h +: 48] = compress_sum + compress_carry;
        end
    end endgenerate
endmodule
`default_nettype wire
