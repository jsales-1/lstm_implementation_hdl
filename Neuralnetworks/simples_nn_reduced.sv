module small_network #(
    parameter int WIDTH = 32,
    parameter int FRAC  = 16,
    parameter int L1_INPUTS  = 2,
    parameter int L1_NEURONS = 4,
    parameter int L2_INPUTS  = 4,
    parameter int L2_NEURONS = 1
)(
    input  logic clk,
    input  logic mode, // 0=write, 1=run

    input  logic we,
    input  logic [20:0] addr,
    input  logic signed [WIDTH-1:0] data_in,

    input  logic signed [WIDTH-1:0] x1 [L1_INPUTS],

    output logic signed [WIDTH-1:0] y2 [L2_NEURONS],
    output logic ready
);

    // =============================
    // MEMÓRIA
    // =============================

    logic [20:0] addr_mux;
    logic signed [WIDTH-1:0] data_out;

    weight_bank wb (
        .clk(clk),
        .we(mode ? 1'b0 : we),     // só escreve no modo WRITE
        .addr(addr_mux),
        .data_in(data_in),
        .data_out(data_out)
    );

    // =============================
    // PESOS INTERNOS
    // =============================

    logic signed [WIDTH-1:0] w1 [L1_NEURONS][L1_INPUTS];
    logic signed [WIDTH-1:0] b1 [L1_NEURONS];

    logic signed [WIDTH-1:0] w2 [L2_NEURONS][L2_INPUTS];
    logic signed [WIDTH-1:0] b2 [L2_NEURONS];

    logic signed [WIDTH-1:0] y1 [L1_NEURONS];

    // =============================
    // FSM
    // =============================

    typedef enum logic [2:0] {
        IDLE,
        LOAD_L1_W,
        LOAD_L1_B,
        LOAD_L2_W,
        LOAD_L2_B,
        RUN
    } state_t;

    state_t state;

    integer n, i;

    // endereço interno
    logic [20:0] addr_internal;

    // mux de endereço
    assign addr_mux = (mode == 0) ? addr : addr_internal;

    // =============================
    // CONTROLE
    // =============================

    always_ff @(posedge clk) begin

        if (mode == 0) begin
            // WRITE MODE
            state <= IDLE;
            ready <= 0;
        end else begin
            // RUN MODE

            case (state)

                IDLE: begin
                    n <= 0;
                    i <= 0;
                    ready <= 0;
                    state <= LOAD_L1_W;
                end

                LOAD_L1_W: begin
                    addr_internal <= {4'd0,1'b0,n[7:0],i[7:0]};
                    w1[n][i] <= data_out;

                    if (i == L1_INPUTS-1) begin
                        i <= 0;
                        if (n == L1_NEURONS-1) begin
                            n <= 0;
                            state <= LOAD_L1_B;
                        end else n <= n + 1;
                    end else i <= i + 1;
                end

                LOAD_L1_B: begin
                    addr_internal <= {4'd0,1'b1,n[7:0],8'd0};
                    b1[n] <= data_out;

                    if (n == L1_NEURONS-1) begin
                        n <= 0;
                        state <= LOAD_L2_W;
                    end else n <= n + 1;
                end

                LOAD_L2_W: begin
                    addr_internal <= {4'd1,1'b0,8'd0,i[7:0]};
                    w2[0][i] <= data_out;

                    if (i == L2_INPUTS-1) begin
                        i <= 0;
                        state <= LOAD_L2_B;
                    end else i <= i + 1;
                end

                LOAD_L2_B: begin
                    addr_internal <= {4'd1,1'b1,8'd0,8'd0};
                    b2[0] <= data_out;
                    state <= RUN;
                end

                RUN: begin
                    ready <= 1;
                end

            endcase
        end
    end

    // =============================
    // CAMADAS
    // =============================

    sigmoid_layer #(
        .N_INPUTS(L1_INPUTS),
        .N_NEURONS(L1_NEURONS),
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) layer1 (
        .x(x1),
        .w(w1),
        .bias(b1),
        .y(y1)
    );

    linear_layer #(
        .N_INPUTS(L2_INPUTS),
        .N_NEURONS(L2_NEURONS),
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) layer2 (
        .x(y1),
        .w(w2),
        .bias(b2),
        .y(y2)
    );

endmodule