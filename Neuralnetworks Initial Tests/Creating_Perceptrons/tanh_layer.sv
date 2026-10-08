module perceptron_tanh #(
    parameter int N     = 4,
    parameter int WIDTH = 32,
    parameter int FRAC  = 16
)(
    input  logic signed [WIDTH-1:0] x [N],
    input  logic signed [WIDTH-1:0] w [N],
    input  logic signed [WIDTH-1:0] bias,

    output logic signed [WIDTH-1:0] y
);

    integer i;

    logic signed [WIDTH-1:0] acc;
    logic signed [WIDTH-1:0] z;

    logic signed [2*WIDTH-1:0] mult;

    logic signed [WIDTH-1:0] z2;
    logic signed [WIDTH-1:0] z3;
    logic signed [WIDTH-1:0] z5;
    logic signed [WIDTH-1:0] z7;
    logic signed [WIDTH-1:0] z9;

    // constantes Q16.16
    localparam signed [WIDTH-1:0] ONE     = 32'sd65536;     // 1.0
    localparam signed [WIDTH-1:0] NEG_ONE = -32'sd65536;    // -1.0
    localparam signed [WIDTH-1:0] ZERO    = 32'sd0;

    // saturação (±3 já é suficiente para tanh)
    localparam signed [WIDTH-1:0] SAT_P  = 32'sd196608;   // +3.0
    localparam signed [WIDTH-1:0] SAT_N  = -32'sd196608;  // -3.0

    always_comb begin

        // soma linear
        acc = bias;

        for (i = 0; i < N; i++) begin
            mult = x[i] * w[i];
            acc  = acc + (mult >>> FRAC);
        end

        z = acc;

        // saturação
        if (z >= SAT_P) begin
            y = ONE;
        end
        else if (z <= SAT_N) begin
            y = NEG_ONE;
        end
        else begin

            // z²
            mult = z * z;
            z2   = mult >>> FRAC;

            // z³
            mult = z2 * z;
            z3   = mult >>> FRAC;

            // z⁵
            mult = z3 * z2;
            z5   = mult >>> FRAC;

            // z⁷
            mult = z5 * z2;
            z7   = mult >>> FRAC;

            // z⁹
            mult = z7 * z2;
            z9   = mult >>> FRAC;

            // tanh Taylor (ordem 9)
            // z - z³/3 + 2z⁵/15 - 17z⁷/315 + 62z⁹/2835

            y =
                  z
                - (z3 / 3)
                + ((2 * z5) / 15)
                - ((17 * z7) / 315)
                + ((62 * z9) / 2835);

            // clamp final
            if (y > ONE)
                y = ONE;
            else if (y < NEG_ONE)
                y = NEG_ONE;
        end

    end

endmodule



module tanh_layer #(
    parameter int N_INPUTS  = 4,
    parameter int N_NEURONS = 3,
    parameter int WIDTH     = 32,
    parameter int FRAC      = 16
)(
    input  logic signed [WIDTH-1:0] x [N_INPUTS],
    input  logic signed [WIDTH-1:0] w [N_NEURONS][N_INPUTS],
    input  logic signed [WIDTH-1:0] bias [N_NEURONS],

    output logic signed [WIDTH-1:0] y [N_NEURONS]
);

    genvar n;

    generate
        for (n = 0; n < N_NEURONS; n++) begin : GEN_NEURONS

            perceptron_tanh #(
                .N(N_INPUTS),
                .WIDTH(WIDTH),
                .FRAC(FRAC)
            ) neuron (
                .x(x),
                .w(w[n]),
                .bias(bias[n]),
                .y(y[n])
            );

        end
    endgenerate

endmodule