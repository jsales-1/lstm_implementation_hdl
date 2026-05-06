module perceptron_sigmoid #(
    parameter int N = 4
)(
    input  shortreal x [N],
    input  shortreal w [N],
    input  shortreal bias,

    output shortreal y
);

    integer i;
    shortreal acc;
    shortreal z;

   // Sigmoid com Taylor mais precisa
//
// sigmoid(z) = 1 / (1 + e^(-z))
//
// Expansão de Taylor em torno de z = 0:
//
// sigmoid(z) ≈
//      1/2
//    + z/4
//    - z^3/48
//    + z^5/480
//    - 17z^7/80640
//
// Muito melhor que a versão anterior.
// Boa precisão em torno de |z| <= 5
//
// Mantemos saturação fora da faixa útil.

always_comb begin

    acc = bias;

    for (i = 0; i < N; i++) begin
        acc = acc + (x[i] * w[i]);
    end

    z = acc;

    // saturação para estabilidade
    if (z <= -5.0) begin
        y = 0.0;
    end
    else if (z >= 5.0) begin
        y = 1.0;
    end
    else begin
        shortreal z2;
        shortreal z3;
        shortreal z5;
        shortreal z7;

        z2 = z * z;
        z3 = z2 * z;
        z5 = z3 * z2;
        z7 = z5 * z2;

        y = 0.5
          + (z / 4.0)
          - (z3 / 48.0)
          + (z5 / 480.0)
          - ((17.0 * z7) / 80640.0);

        // clamp de segurança
        if (y < 0.0)
            y = 0.0;
        else if (y > 1.0)
            y = 1.0;
    end

end

endmodule

module sigmoid_layer #(
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

            perceptron_sigmoid #(
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