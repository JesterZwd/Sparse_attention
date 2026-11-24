// Page-header coefficients for the Round 12 compact scorers.
`default_nettype none

module r12_q24_coeff (
    input wire clk, rst_n, in_valid,
    input wire [63:0] g64,
    input wire [15:0] scale,
    input wire radius,
    output wire out_valid,
    output reg [21:0] coeff
);
    // Each input is accepted independently. An input at edge n produces
    // its output just after edge n+8. A synchronous reset clears all valids.
    reg [8:0] valid;
    assign out_valid = valid[8];

    function automatic [3:0] half_leading_zeros(input [10:0] significand);
        casez (significand)
            11'b1??????????: half_leading_zeros = 4'd0;
            11'b01?????????: half_leading_zeros = 4'd1;
            11'b001????????: half_leading_zeros = 4'd2;
            11'b0001???????: half_leading_zeros = 4'd3;
            11'b00001??????: half_leading_zeros = 4'd4;
            11'b000001?????: half_leading_zeros = 4'd5;
            11'b0000001????: half_leading_zeros = 4'd6;
            11'b00000001???: half_leading_zeros = 4'd7;
            11'b000000001??: half_leading_zeros = 4'd8;
            11'b0000000001?: half_leading_zeros = 4'd9;
            default: half_leading_zeros = 4'd10;
        endcase
    endfunction

    wire [10:0] gain_exp = g64[62:52];
    wire [4:0] half_exp = scale[14:10];
    wire [4:0] half_exp_adjusted = half_exp == 0 ? 5'd1 : half_exp;
    wire [10:0] half_sig = {half_exp != 0, scale[9:0]};
    wire [3:0] half_lz = half_leading_zeros(half_sig);

    reg [52:0] gain_sig1;
    reg [10:0] half_sig1;
    reg [11:0] exp_sum1;
    reg [3:0] half_lz1;
    reg radius1, zero1;

    reg [23:0] pp0, pp1, pp2;
    reg [24:0] pp3;
    reg signed [11:0] shift2, shift3, shift4;
    reg zero2, zero3, zero4;
    reg [36:0] product_lo3;
    reg [37:0] product_hi3;
    reg [63:0] product4;
    wire [37:0] product_hi = product_hi3 + {27'b0, product_lo3[36:26]};

    wire signed [11:0] window_shift = shift4 - $signed({11'b0, product4[63]});
    wire [52:0] fp_trunc = product4[63] ? product4[63:11] : product4[62:10];
    wire fp_guard = product4[63] ? product4[10] : product4[9];
    wire fp_sticky = (|product4[8:0]) | (product4[63] & product4[9]);
    reg [4:0] shift5;
    reg [2:0] shift6;
    reg [24:0] window5, window6, window7;
    reg fp_increment5, fp_increment6, fp_increment7;
    reg small5, small6, small7, small8;
    reg large5, large6, large7, large8;

    // The shifter has a 22-bit integer, one guard bit, and two predicates
    // for the already-discarded bits (any one bit and all one bits).
    wire [24:0] window_in [0:4];
    wire [24:0] window_out [0:4];
    assign window_in[0] = window5;
    assign window_in[1] = window_out[0];
    assign window_in[2] = window6;
    assign window_in[3] = window_out[2];
    assign window_in[4] = window_out[3];
    genvar step;
    generate for (step=0; step<5; step=step+1) begin : right_shift
        localparam DISTANCE = 1 << (4-step);
        wire take_step;
        if (step < 2) assign take_step = shift5[4-step];
        else          assign take_step = shift6[4-step];
        wire [24:0] w = window_in[step];
        wire [22:0] shifted = w[24:2] >> DISTANCE;
        wire sticky = w[1] | (|w[DISTANCE+1:2]);
        wire all_ones = w[0] & (&w[DISTANCE+1:2]);
        assign window_out[step] = take_step ? {shifted, sticky, all_ones} : w;
    end endgenerate

    wire [21:0] integer_part = window7[24:3];
    wire guard = window7[2];
    wire sticky = window7[1];
    wire all_ones = window7[0];
    // The preceding binary64 increment is at the bottom of the discarded
    // region. With guard=1 it can only increase the rounding decision.
    // With guard=0 it can reach a tie only when every lower bit is set.
    wire round_up = guard ? sticky | fp_increment7 | integer_part[0]
                          : fp_increment7 & all_ones & integer_part[0];
    reg [21:0] integer8;
    reg increment8;

    always @(posedge clk) begin
        if (!rst_n)
            valid <= 9'b0;
        else
            valid <= {valid[7:0], in_valid};

        if (in_valid) begin
            gain_sig1 <= {1'b1, g64[51:0]};
            half_sig1 <= half_sig << half_lz;
            exp_sum1 <= {1'b0, gain_exp} + {7'b0, half_exp_adjusted};
            half_lz1 <= half_lz;
            radius1 <= radius;
            // A binary64 subnormal cannot reach half a Q24 unit with any
            // finite half. Negative or nonfinite encodings also produce zero.
            zero1 <= g64[63] | scale[15] | (gain_exp == 0) | (&gain_exp) |
                     (&half_exp) | (half_sig == 0);
        end

        pp0 <= gain_sig1[12:0] * half_sig1;
        pp1 <= gain_sig1[25:13] * half_sig1;
        pp2 <= gain_sig1[38:26] * half_sig1;
        pp3 <= gain_sig1[52:39] * half_sig1;
        // Exponent offset: 1075+25 for the significands, minus 24 for
        // Q24, minus 10 or 11 for the binary64 product, minus 31 for the
        // fixed portion of the integer shift. Radius adds a division by 4.
        shift2 <= 12'sd1035 - exp_sum1 + {8'b0, half_lz1} + {10'b0, radius1, 1'b0};
        zero2 <= zero1;

        product_lo3 <= {pp1, 13'b0} + {13'b0, pp0};
        product_hi3 <= {pp3, 13'b0} + {14'b0, pp2};
        shift3 <= shift2;
        zero3 <= zero2;

        product4 <= {product_hi, product_lo3[25:0]};
        shift4 <= shift3;
        zero4 <= zero3;

        window5 <= {fp_trunc[52:30], |fp_trunc[29:0], &fp_trunc[29:0]};
        fp_increment5 <= fp_guard & (fp_sticky | fp_trunc[0]);
        shift5 <= window_shift[4:0];
        // A normalized product has 53 bits. A shift of 30 or less
        // saturates U22. A shift of 54 or more rounds to zero, even
        // when the binary64 increment grows the mantissa by one bit.
        small5 <= zero4 | (window_shift > 12'sd22);
        large5 <= window_shift < 12'sd0;

        window6 <= window_out[1];
        shift6 <= shift5[2:0];
        fp_increment6 <= fp_increment5;
        small6 <= small5;
        large6 <= large5;

        window7 <= window_out[4];
        fp_increment7 <= fp_increment6;
        small7 <= small6;
        large7 <= large6;

        integer8 <= integer_part;
        increment8 <= round_up & ~(&integer_part);
        small8 <= small7;
        large8 <= large7;

        if (small8)
            coeff <= 22'b0;
        else if (large8)
            coeff <= 22'h3fffff;
        else
            coeff <= integer8 + increment8;
    end
endmodule


module r12_length_bias (
    input wire [7:0] count,
    output reg signed [31:0] bias
);
    // RNE(2**24 * ln(count)) for a 32-token page. The invalid count 0
    // and out-of-range header encodings have zero bias.
    always @* begin
        case (count)
            8'd2: bias = 32'sd11629080;
            8'd3: bias = 32'sd18431656;
            8'd4: bias = 32'sd23258160;
            8'd5: bias = 32'sd27001887;
            8'd6: bias = 32'sd30060736;
            8'd7: bias = 32'sd32646955;
            8'd8: bias = 32'sd34887240;
            8'd9: bias = 32'sd36863311;
            8'd10: bias = 32'sd38630967;
            8'd11: bias = 32'sd40230007;
            8'd12: bias = 32'sd41689816;
            8'd13: bias = 32'sd43032709;
            8'd14: bias = 32'sd44276035;
            8'd15: bias = 32'sd45433543;
            8'd16: bias = 32'sd46516320;
            8'd17: bias = 32'sd47533432;
            8'd18: bias = 32'sd48492391;
            8'd19: bias = 32'sd49399489;
            8'd20: bias = 32'sd50260047;
            8'd21: bias = 32'sd51078611;
            8'd22: bias = 32'sd51859087;
            8'd23: bias = 32'sd52604864;
            8'd24: bias = 32'sd53318896;
            8'd25: bias = 32'sd54003775;
            8'd26: bias = 32'sd54661789;
            8'd27: bias = 32'sd55294967;
            8'd28: bias = 32'sd55905115;
            8'd29: bias = 32'sd56493849;
            8'd30: bias = 32'sd57062623;
            8'd31: bias = 32'sd57612745;
            8'd32: bias = 32'sd58145400;
            default: bias = 32'sd0;
        endcase
    end
endmodule

`default_nettype wire
