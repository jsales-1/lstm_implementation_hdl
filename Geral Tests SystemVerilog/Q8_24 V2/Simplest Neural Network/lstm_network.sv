module lstm_network #(
    parameter int WIDTH = 32,
    parameter int FRAC  = 24,
    
    parameter int LSTM_INPUTS  = 4,
    parameter int LSTM_HIDDEN  = 2,        // Máximo 8 (3 bits)
    parameter int TIMESTEPS    = 5,
    parameter int RELU_INPUTS  = 2,         // Máximo 8 (3 bits)
    parameter int RELU_NEURONS = 4,         // Máximo 8 (3 bits)
    parameter int OUT_INPUTS   = 4          // Máximo 8 (3 bits)
)(
    input  logic clk,
    input  logic reset,
    input  logic mode,
    input  logic clear,
    input  logic we,
    input  logic [11:0] addr,              // 12 bits (era 21)
    input  logic signed [WIDTH-1:0] data_in,
    input  logic signed [WIDTH-1:0] x [TIMESTEPS][LSTM_INPUTS],
    output logic signed [WIDTH-1:0] y_out,
    output logic ready
);

    logic [11:0] addr_mux;
    logic [11:0] addr_internal;
    logic signed [WIDTH-1:0] data_out;

    weight_bank #(.WIDTH(WIDTH), .MAX_SIZE(4096)) wb (
        .clk(clk),
        .rst(clear),
        .we(mode ? 1'b0 : we),
        .addr(addr_mux),
        .data_in(data_in),
        .data_out(data_out)
    );

    assign addr_mux = (mode == 0) ? addr : addr_internal;

    // ============================================================
    // PESOS LSTM (limitados para 12 bits)
    // ============================================================
    // LSTM_HIDDEN = 8, LSTM_INPUTS = 4
    
    // Forget Gate (gate 01 - o que funciona)
    logic signed [WIDTH-1:0] lstm_wx_forget [LSTM_HIDDEN][LSTM_INPUTS];
    logic signed [WIDTH-1:0] lstm_wh_forget [LSTM_HIDDEN][LSTM_HIDDEN];
    logic signed [WIDTH-1:0] lstm_bias_forget [LSTM_HIDDEN];
    
    // Input Gate (gate 00 - o que funciona)
    logic signed [WIDTH-1:0] lstm_wx_input [LSTM_HIDDEN][LSTM_INPUTS];
    logic signed [WIDTH-1:0] lstm_wh_input [LSTM_HIDDEN][LSTM_HIDDEN];
    logic signed [WIDTH-1:0] lstm_bias_input [LSTM_HIDDEN];
    
    // Candidate Gate (gate 10)
    logic signed [WIDTH-1:0] lstm_wx_cell [LSTM_HIDDEN][LSTM_INPUTS];
    logic signed [WIDTH-1:0] lstm_wh_cell [LSTM_HIDDEN][LSTM_HIDDEN];
    logic signed [WIDTH-1:0] lstm_bias_cell [LSTM_HIDDEN];
    
    // Output Gate (gate 11)
    logic signed [WIDTH-1:0] lstm_wx_output [LSTM_HIDDEN][LSTM_INPUTS];
    logic signed [WIDTH-1:0] lstm_wh_output [LSTM_HIDDEN][LSTM_HIDDEN];
    logic signed [WIDTH-1:0] lstm_bias_output [LSTM_HIDDEN];
    
    // ============================================================
    // PESOS ReLU (layer 1)
    // ============================================================
    logic signed [WIDTH-1:0] relu_w [RELU_NEURONS][RELU_INPUTS];
    logic signed [WIDTH-1:0] relu_bias [RELU_NEURONS];
    
    // ============================================================
    // PESOS Output (layer 2)
    // ============================================================
    logic signed [WIDTH-1:0] out_w [1][OUT_INPUTS];
    logic signed [WIDTH-1:0] out_bias [1];
    
    // ============================================================
    // SINAIS INTERMEDIÁRIOS
    // ============================================================
    logic signed [WIDTH-1:0] lstm_h_out [TIMESTEPS][LSTM_HIDDEN];
    logic signed [WIDTH-1:0] lstm_c_final [LSTM_HIDDEN];
    logic signed [WIDTH-1:0] relu_y [RELU_NEURONS];
    logic signed [WIDTH-1:0] out_temp [1];
    
    logic lstm_done;
    logic lstm_start;

    // ============================================================
    // FSM PARA CARREGAR PESOS (12 bits)
    // ============================================================
    // Bits 11-10: layer     (2 bits) → 0=LSTM, 1=ReLU, 2=Output
    // Bit  9:     is_bias   (1 bit)  → 0=peso, 1=bias
    // Bits 8-7:   gate      (2 bits) → 00=input, 01=forget, 10=candidate, 11=output
    // Bits 6-4:   neuron    (3 bits) → 0 a 7
    // Bit  3:     recurrent (1 bit)  → 0=Wx, 1=Wh
    // Bits 2-0:   idx       (3 bits) → 0 a 7

    typedef enum logic [5:0] {
        IDLE,
        LOAD_LSTM_WX_FORGET, LOAD_LSTM_WX_FORGET_DATA,
        LOAD_LSTM_WH_FORGET, LOAD_LSTM_WH_FORGET_DATA,
        LOAD_LSTM_BIAS_FORGET, LOAD_LSTM_BIAS_FORGET_DATA,
        LOAD_LSTM_WX_INPUT, LOAD_LSTM_WX_INPUT_DATA,
        LOAD_LSTM_WH_INPUT, LOAD_LSTM_WH_INPUT_DATA,
        LOAD_LSTM_BIAS_INPUT, LOAD_LSTM_BIAS_INPUT_DATA,
        LOAD_LSTM_WX_CELL, LOAD_LSTM_WX_CELL_DATA,
        LOAD_LSTM_WH_CELL, LOAD_LSTM_WH_CELL_DATA,
        LOAD_LSTM_BIAS_CELL, LOAD_LSTM_BIAS_CELL_DATA,
        LOAD_LSTM_WX_OUTPUT, LOAD_LSTM_WX_OUTPUT_DATA,
        LOAD_LSTM_WH_OUTPUT, LOAD_LSTM_WH_OUTPUT_DATA,
        LOAD_LSTM_BIAS_OUTPUT, LOAD_LSTM_BIAS_OUTPUT_DATA,
        LOAD_RELU_W, LOAD_RELU_W_DATA,
        LOAD_RELU_B, LOAD_RELU_B_DATA,
        LOAD_OUT_W, LOAD_OUT_W_DATA,
        LOAD_OUT_B, LOAD_OUT_B_DATA,
        RUN_LSTM, WAIT_LSTM, RUN_DONE
    } state_t;

    state_t state;
    logic [2:0] neuron;    // 3 bits (0 a 7)
    logic [2:0] idx;       // 3 bits (0 a 7)

    // ============================================================
    // FUNÇÃO DE ENDEREÇO (12 bits)
    // ============================================================
    function logic [11:0] gen_addr(
        input logic [1:0] layer,
        input logic is_bias,
        input logic [1:0] gate,
        input logic [2:0] neuron,
        input logic recurrent,
        input logic [2:0] idx
    );
        gen_addr = {layer, is_bias, gate, neuron, recurrent, idx};
    endfunction

    // ============================================================
    // START SIGNAL
    // ============================================================
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            lstm_start <= 1'b0;
        end else begin
            if (state == RUN_LSTM) begin
                lstm_start <= 1'b1;
            end else if (lstm_done) begin
                lstm_start <= 1'b0;
            end
        end
    end

    // ============================================================
    // FSM PRINCIPAL (12 bits)
    // ============================================================
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            ready <= 1'b0;
            neuron <= 0;
            idx <= 0;
            addr_internal <= 0;
        end
        else if (mode == 0) begin
            state <= IDLE;
            ready <= 1'b0;
            neuron <= 0;
            idx <= 0;
            addr_internal <= 0;
        end
        else begin
            case (state)
                IDLE: begin
                    ready <= 1'b0;
                    neuron <= 0;
                    idx <= 0;
                    state <= LOAD_LSTM_WX_FORGET;
                end

                // ============================================================
                // FORGET GATE (gate 01 - o que funciona)
                // ============================================================
                LOAD_LSTM_WX_FORGET: begin
                    addr_internal <= gen_addr(2'b00, 1'b0, 2'b01, neuron, 1'b0, idx);
                    state <= LOAD_LSTM_WX_FORGET_DATA;
                end
                
                LOAD_LSTM_WX_FORGET_DATA: begin
                    lstm_wx_forget[neuron][idx] <= data_out;
                    if (idx == LSTM_INPUTS-1) begin
                        idx <= 0;
                        if (neuron == LSTM_HIDDEN-1) begin
                            neuron <= 0;
                            state <= LOAD_LSTM_WH_FORGET;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_LSTM_WX_FORGET;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_LSTM_WX_FORGET;
                    end
                end
                
                LOAD_LSTM_WH_FORGET: begin
                    addr_internal <= gen_addr(2'b00, 1'b0, 2'b01, neuron, 1'b1, idx);
                    state <= LOAD_LSTM_WH_FORGET_DATA;
                end
                
                LOAD_LSTM_WH_FORGET_DATA: begin
                    lstm_wh_forget[neuron][idx] <= data_out;
                    if (idx == LSTM_HIDDEN-1) begin
                        idx <= 0;
                        if (neuron == LSTM_HIDDEN-1) begin
                            neuron <= 0;
                            state <= LOAD_LSTM_BIAS_FORGET;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_LSTM_WH_FORGET;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_LSTM_WH_FORGET;
                    end
                end
                
                LOAD_LSTM_BIAS_FORGET: begin
                    addr_internal <= gen_addr(2'b00, 1'b1, 2'b01, neuron, 1'b0, 3'b0);
                    state <= LOAD_LSTM_BIAS_FORGET_DATA;
                end
                
                LOAD_LSTM_BIAS_FORGET_DATA: begin
                    lstm_bias_forget[neuron] <= data_out;
                    if (neuron == LSTM_HIDDEN-1) begin
                        neuron <= 0;
                        state <= LOAD_LSTM_WX_INPUT;
                    end else begin
                        neuron <= neuron + 1;
                        state <= LOAD_LSTM_BIAS_FORGET;
                    end
                end

                // ============================================================
                // INPUT GATE (gate 00 - o que funciona)
                // ============================================================
                LOAD_LSTM_WX_INPUT: begin
                    addr_internal <= gen_addr(2'b00, 1'b0, 2'b00, neuron, 1'b0, idx);
                    state <= LOAD_LSTM_WX_INPUT_DATA;
                end
                
                LOAD_LSTM_WX_INPUT_DATA: begin
                    lstm_wx_input[neuron][idx] <= data_out;
                    if (idx == LSTM_INPUTS-1) begin
                        idx <= 0;
                        if (neuron == LSTM_HIDDEN-1) begin
                            neuron <= 0;
                            state <= LOAD_LSTM_WH_INPUT;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_LSTM_WX_INPUT;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_LSTM_WX_INPUT;
                    end
                end
                
                LOAD_LSTM_WH_INPUT: begin
                    addr_internal <= gen_addr(2'b00, 1'b0, 2'b00, neuron, 1'b1, idx);
                    state <= LOAD_LSTM_WH_INPUT_DATA;
                end
                
                LOAD_LSTM_WH_INPUT_DATA: begin
                    lstm_wh_input[neuron][idx] <= data_out;
                    if (idx == LSTM_HIDDEN-1) begin
                        idx <= 0;
                        if (neuron == LSTM_HIDDEN-1) begin
                            neuron <= 0;
                            state <= LOAD_LSTM_BIAS_INPUT;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_LSTM_WH_INPUT;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_LSTM_WH_INPUT;
                    end
                end
                
                LOAD_LSTM_BIAS_INPUT: begin
                    addr_internal <= gen_addr(2'b00, 1'b1, 2'b00, neuron, 1'b0, 3'b0);
                    state <= LOAD_LSTM_BIAS_INPUT_DATA;
                end
                
                LOAD_LSTM_BIAS_INPUT_DATA: begin
                    lstm_bias_input[neuron] <= data_out;
                    if (neuron == LSTM_HIDDEN-1) begin
                        neuron <= 0;
                        state <= LOAD_LSTM_WX_CELL;
                    end else begin
                        neuron <= neuron + 1;
                        state <= LOAD_LSTM_BIAS_INPUT;
                    end
                end

                // ============================================================
                // CANDIDATE GATE (gate 10)
                // ============================================================
                LOAD_LSTM_WX_CELL: begin
                    addr_internal <= gen_addr(2'b00, 1'b0, 2'b10, neuron, 1'b0, idx);
                    state <= LOAD_LSTM_WX_CELL_DATA;
                end
                
                LOAD_LSTM_WX_CELL_DATA: begin
                    lstm_wx_cell[neuron][idx] <= data_out;
                    if (idx == LSTM_INPUTS-1) begin
                        idx <= 0;
                        if (neuron == LSTM_HIDDEN-1) begin
                            neuron <= 0;
                            state <= LOAD_LSTM_WH_CELL;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_LSTM_WX_CELL;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_LSTM_WX_CELL;
                    end
                end
                
                LOAD_LSTM_WH_CELL: begin
                    addr_internal <= gen_addr(2'b00, 1'b0, 2'b10, neuron, 1'b1, idx);
                    state <= LOAD_LSTM_WH_CELL_DATA;
                end
                
                LOAD_LSTM_WH_CELL_DATA: begin
                    lstm_wh_cell[neuron][idx] <= data_out;
                    if (idx == LSTM_HIDDEN-1) begin
                        idx <= 0;
                        if (neuron == LSTM_HIDDEN-1) begin
                            neuron <= 0;
                            state <= LOAD_LSTM_BIAS_CELL;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_LSTM_WH_CELL;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_LSTM_WH_CELL;
                    end
                end
                
                LOAD_LSTM_BIAS_CELL: begin
                    addr_internal <= gen_addr(2'b00, 1'b1, 2'b10, neuron, 1'b0, 3'b0);
                    state <= LOAD_LSTM_BIAS_CELL_DATA;
                end
                
                LOAD_LSTM_BIAS_CELL_DATA: begin
                    lstm_bias_cell[neuron] <= data_out;
                    if (neuron == LSTM_HIDDEN-1) begin
                        neuron <= 0;
                        state <= LOAD_LSTM_WX_OUTPUT;
                    end else begin
                        neuron <= neuron + 1;
                        state <= LOAD_LSTM_BIAS_CELL;
                    end
                end

                // ============================================================
                // OUTPUT GATE (gate 11)
                // ============================================================
                LOAD_LSTM_WX_OUTPUT: begin
                    addr_internal <= gen_addr(2'b00, 1'b0, 2'b11, neuron, 1'b0, idx);
                    state <= LOAD_LSTM_WX_OUTPUT_DATA;
                end
                
                LOAD_LSTM_WX_OUTPUT_DATA: begin
                    lstm_wx_output[neuron][idx] <= data_out;
                    if (idx == LSTM_INPUTS-1) begin
                        idx <= 0;
                        if (neuron == LSTM_HIDDEN-1) begin
                            neuron <= 0;
                            state <= LOAD_LSTM_WH_OUTPUT;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_LSTM_WX_OUTPUT;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_LSTM_WX_OUTPUT;
                    end
                end
                
                LOAD_LSTM_WH_OUTPUT: begin
                    addr_internal <= gen_addr(2'b00, 1'b0, 2'b11, neuron, 1'b1, idx);
                    state <= LOAD_LSTM_WH_OUTPUT_DATA;
                end
                
                LOAD_LSTM_WH_OUTPUT_DATA: begin
                    lstm_wh_output[neuron][idx] <= data_out;
                    if (idx == LSTM_HIDDEN-1) begin
                        idx <= 0;
                        if (neuron == LSTM_HIDDEN-1) begin
                            neuron <= 0;
                            state <= LOAD_LSTM_BIAS_OUTPUT;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_LSTM_WH_OUTPUT;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_LSTM_WH_OUTPUT;
                    end
                end
                
                LOAD_LSTM_BIAS_OUTPUT: begin
                    addr_internal <= gen_addr(2'b00, 1'b1, 2'b11, neuron, 1'b0, 3'b0);
                    state <= LOAD_LSTM_BIAS_OUTPUT_DATA;
                end
                
                LOAD_LSTM_BIAS_OUTPUT_DATA: begin
                    lstm_bias_output[neuron] <= data_out;
                    if (neuron == LSTM_HIDDEN-1) begin
                        neuron <= 0;
                        state <= LOAD_RELU_W;
                    end else begin
                        neuron <= neuron + 1;
                        state <= LOAD_LSTM_BIAS_OUTPUT;
                    end
                end

                // ============================================================
                // ReLU WEIGHTS (layer = 1)
                // ============================================================
                LOAD_RELU_W: begin
                    addr_internal <= gen_addr(2'b01, 1'b0, 2'b00, neuron, 1'b0, idx);
                    state <= LOAD_RELU_W_DATA;
                end
                
                LOAD_RELU_W_DATA: begin
                    relu_w[neuron][idx] <= data_out;
                    if (idx == RELU_INPUTS-1) begin
                        idx <= 0;
                        if (neuron == RELU_NEURONS-1) begin
                            neuron <= 0;
                            state <= LOAD_RELU_B;
                        end else begin
                            neuron <= neuron + 1;
                            state <= LOAD_RELU_W;
                        end
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_RELU_W;
                    end
                end
                
                LOAD_RELU_B: begin
                    addr_internal <= gen_addr(2'b01, 1'b1, 2'b00, neuron, 1'b0, 3'b0);
                    state <= LOAD_RELU_B_DATA;
                end
                
                LOAD_RELU_B_DATA: begin
                    relu_bias[neuron] <= data_out;
                    if (neuron == RELU_NEURONS-1) begin
                        neuron <= 0;
                        state <= LOAD_OUT_W;
                    end else begin
                        neuron <= neuron + 1;
                        state <= LOAD_RELU_B;
                    end
                end

                // ============================================================
                // OUTPUT WEIGHTS (layer = 2)
                // ============================================================
                LOAD_OUT_W: begin
                    addr_internal <= gen_addr(2'b10, 1'b0, 2'b00, 3'b0, 1'b0, idx);
                    state <= LOAD_OUT_W_DATA;
                end
                
                LOAD_OUT_W_DATA: begin
                    out_w[0][idx] <= data_out;
                    if (idx == OUT_INPUTS-1) begin
                        idx <= 0;
                        state <= LOAD_OUT_B;
                    end else begin
                        idx <= idx + 1;
                        state <= LOAD_OUT_W;
                    end
                end
                
                LOAD_OUT_B: begin
                    addr_internal <= gen_addr(2'b10, 1'b1, 2'b00, 3'b0, 1'b0, 3'b0);
                    state <= LOAD_OUT_B_DATA;
                end
                
                LOAD_OUT_B_DATA: begin
                    out_bias[0] <= data_out;
                    state <= RUN_LSTM;
                end

                // ============================================================
                // EXECUTA LSTM
                // ============================================================
                RUN_LSTM: begin
                    ready <= 1'b0;
                    if (lstm_done) begin
                        state <= WAIT_LSTM;
                    end
                end
                
                WAIT_LSTM: begin
                    ready <= 1'b0;
                    if (lstm_done) begin
                        state <= RUN_DONE;
                    end
                end
                
                RUN_DONE: begin
                    ready <= 1'b1;
                end
            endcase
        end
    end

    // ============================================================
    // INSTÂNCIAS DAS CAMADAS
    // ============================================================
    
    // Camada 1: LSTM
    lstm_layer #(
        .N_INPUTS(LSTM_INPUTS),
        .N_HIDDEN(LSTM_HIDDEN),
        .N_TIMESTEPS(TIMESTEPS),
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) lstm_inst (
        .clk(clk),
        .rst_n(~reset),
        .start(lstm_start),
        .x(x),
        .w_ix(lstm_wx_input),
        .w_ih(lstm_wh_input),
        .w_fx(lstm_wx_forget),
        .w_fh(lstm_wh_forget),
        .w_ox(lstm_wx_output),
        .w_oh(lstm_wh_output),
        .w_cx(lstm_wx_cell),
        .w_ch(lstm_wh_cell),
        .bias_i(lstm_bias_input),
        .bias_f(lstm_bias_forget),
        .bias_o(lstm_bias_output),
        .bias_c(lstm_bias_cell),
        .h_out(lstm_h_out),
        .c_final(lstm_c_final),
        .done(lstm_done)
    );
    
    // Camada 2: ReLU
    relu_layer #(
        .N_INPUTS(RELU_INPUTS),
        .N_NEURONS(RELU_NEURONS),
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) relu_inst (
        .x(lstm_h_out[TIMESTEPS-1]),
        .w(relu_w),
        .bias(relu_bias),
        .y(relu_y)
    );
    
    // Camada 3: Sigmoid Output
    sigmoid_layer #(
        .N_INPUTS(OUT_INPUTS),
        .N_NEURONS(1),
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) out_inst (
        .x(relu_y),
        .w(out_w),
        .bias(out_bias),
        .y(out_temp)
    );
    
    assign y_out = out_temp[0];

endmodule