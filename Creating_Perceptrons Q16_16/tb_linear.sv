`timescale 1ns/1ps

module tb_linear_network;

    // =====================================================
    // FLOAT32 / shortreal
    // =====================================================

    // =====================================================
    // CAMADA 1
    // 2 entradas -> 2 neurônios
    // =====================================================

    parameter int L1_INPUTS  = 2;
    parameter int L1_NEURONS = 2;

    shortreal x1 [L1_INPUTS];
    shortreal w1 [L1_NEURONS][L1_INPUTS];
    shortreal b1 [L1_NEURONS];
    shortreal y1 [L1_NEURONS];

    // =====================================================
    // CAMADA 2
    // 2 entradas -> 5 neurônios
    // =====================================================

    parameter int L2_INPUTS  = 2;
    parameter int L2_NEURONS = 5;

    shortreal w2 [L2_NEURONS][L2_INPUTS];
    shortreal b2 [L2_NEURONS];
    shortreal y2 [L2_NEURONS];

    // =====================================================
    // CAMADA 3
    // 5 entradas -> 1 neurônio
    // =====================================================

    parameter int L3_INPUTS  = 5;
    parameter int L3_NEURONS = 1;

    shortreal w3 [L3_NEURONS][L3_INPUTS];
    shortreal b3 [L3_NEURONS];
    shortreal y3 [L3_NEURONS];

    // =====================================================
    // DUTs
    // =====================================================

    sigmoid_layer #(
        .N_INPUTS(L1_INPUTS),
        .N_NEURONS(L1_NEURONS)
    ) layer1 (
        .x(x1),
        .w(w1),
        .bias(b1),
        .y(y1)
    );

    linear_layer #(
        .N_INPUTS(L2_INPUTS),
        .N_NEURONS(L2_NEURONS)
    ) layer2 (
        .x(y1),
        .w(w2),
        .bias(b2),
        .y(y2)
    );

    linear_layer #(
        .N_INPUTS(L3_INPUTS),
        .N_NEURONS(L3_NEURONS)
    ) layer3 (
        .x(y2),
        .w(w3),
        .bias(b3),
        .y(y3)
    );

    integer i;

    initial begin

        // =====================================================
        // INPUT EXEMPLO
        // =====================================================

        x1[0] = 1.0;
        x1[1] = 0.5;
        // =====================================================
        // CAMADA 1 - PESOS
        // SHAPE ORIGINAL KERAS: (2, 2)
        // reorganizado para [neurônio][entrada]
        // formato: float32 / shortreal
        // =====================================================

        // neurônio 0
        w1[0][0] = -0.33571896;
        w1[0][1] = -0.60317767;

        // neurônio 1
        w1[1][0] = 0.45532194;
        w1[1][1] = 0.50007170;

        // =====================================================
        // CAMADA 1 - BIAS
        // formato: float32 / shortreal
        // =====================================================

        b1[0] = 0.01223405;
        b1[1] = 0.04571380;

        // =====================================================
        // CAMADA 2 - PESOS
        // SHAPE ORIGINAL KERAS: (2, 5)
        // reorganizado para [neurônio][entrada]
        // formato: float32 / shortreal
        // =====================================================

        // neurônio 0
        w2[0][0] = -2.85537672;
        w2[0][1] = 2.88178754;

        // neurônio 1
        w2[1][0] = 2.53240418;
        w2[1][1] = -3.17465806;

        // neurônio 2
        w2[2][0] = 2.64733005;
        w2[2][1] = -2.96853232;

        // neurônio 3
        w2[3][0] = 2.34361863;
        w2[3][1] = -2.80582309;

        // neurônio 4
        w2[4][0] = 2.73379159;
        w2[4][1] = -2.93898320;

        // =====================================================
        // CAMADA 2 - BIAS
        // formato: float32 / shortreal
        // =====================================================

        b2[0] = -0.22210407;
        b2[1] = 0.22110091;
        b2[2] = 0.20711946;
        b2[3] = 0.23878743;
        b2[4] = 0.21441998;

        // =====================================================
        // CAMADA 3 - PESOS
        // SHAPE ORIGINAL KERAS: (5, 1)
        // reorganizado para [neurônio][entrada]
        // formato: float32 / shortreal
        // =====================================================

        // neurônio 0
        w3[0][0] = 3.20880556;
        w3[0][1] = -3.14888358;
        w3[0][2] = -2.68052459;
        w3[0][3] = -2.90795064;
        w3[0][4] = -3.35732985;

        // =====================================================
        // CAMADA 3 - BIAS
        // formato: float32 / shortreal
        // =====================================================

        b3[0] = -0.16688374;


        // tempo para propagação
        #10;

        // =====================================================
        // DISPLAY
        // =====================================================

        $display("\n==============================");
        $display("ENTRADAS");
        $display("==============================");

        for (i = 0; i < L1_INPUTS; i++) begin
            $display("x1[%0d] = %f", i, x1[i]);
        end

        $display("\n==============================");
        $display("SAIDA CAMADA 1");
        $display("==============================");

        for (i = 0; i < L1_NEURONS; i++) begin
            $display("y1[%0d] = %f", i, y1[i]);
        end

        $display("\n==============================");
        $display("SAIDA CAMADA 2");
        $display("==============================");

        for (i = 0; i < L2_NEURONS; i++) begin
            $display("y2[%0d] = %f", i, y2[i]);
        end

        $display("\n==============================");
        $display("SAIDA FINAL");
        $display("==============================");

        $display("y3[0] = %f", y3[0]);

        $display("\n==============================\n");

        $finish;
    end

endmodule