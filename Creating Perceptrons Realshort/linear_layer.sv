module perceptron_linear #(
    parameter int N = 4
)(
    input  shortreal x [N],
    input  shortreal w [N],
    input  shortreal bias,

    output shortreal y
);

    integer i;
    shortreal acc;

    always_comb begin
        acc = bias;

        for (i = 0; i < N; i++) begin
            acc = acc + (x[i] * w[i]);
        end

        y = acc;
    end

endmodule


module linear_layer #(
    parameter int N_INPUTS   = 4,
    parameter int N_NEURONS  = 3
)(
    input  shortreal x [N_INPUTS],

    input  shortreal w [N_NEURONS][N_INPUTS],

    input  shortreal bias [N_NEURONS],

    output shortreal y [N_NEURONS]
);

    genvar n;

    generate
        for (n = 0; n < N_NEURONS; n++) begin : GEN_NEURONS

            perceptron_linear #(
                .N(N_INPUTS)
            ) neuron (
                .x(x),
                .w(w[n]),
                .bias(bias[n]),
                .y(y[n])
            );

        end
    endgenerate

endmodule
