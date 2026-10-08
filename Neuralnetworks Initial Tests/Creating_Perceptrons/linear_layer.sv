
module perceptron_linear #(
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
    logic signed [2*WIDTH-1:0] mult;

    always_comb begin
        acc = bias;

        for (i = 0; i < N; i++) begin
            mult = x[i] * w[i];

            // ajuste Q16.16:
            // Q16.16 * Q16.16 = Q32.32
            // shift >> 16 para voltar para Q16.16
            acc = acc + (mult >>> FRAC);
        end

        y = acc;
    end

endmodule


module linear_layer #(
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

            perceptron_linear #(
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
