module lstm_network #(
    parameter int WIDTH = 32,
    parameter int FRAC  = 24,
    
    parameter int LSTM_INPUTS  = 4,
    parameter int LSTM_HIDDEN  = 2,
    parameter int TIMESTEPS    = 5,
    parameter int RELU_INPUTS  = 2,
    parameter int RELU_NEURONS = 4,
    parameter int OUT_INPUTS   = 4
)(
    input  logic clk,
    input  logic reset,
    input  logic mode,
    input  logic we,
    input  logic [20:0] addr,
    input  logic signed [WIDTH-1:0] data_in,
    input  logic signed [WIDTH-1:0] x [TIMESTEPS][LSTM_INPUTS],
    output logic signed [WIDTH-1:0] y_out,
    output logic ready
);

    logic [20:0] addr_mux;
    logic [20:0] addr_internal;
    logic signed [WIDTH-1:0] data_out;

    weight_bank wb (
        .clk(clk),
        .we(mode ? 1'b0 : we),
        .addr(addr_mux),
        .data_in(data_in),
        .data_out(data_out)
    );

    assign addr_mux = (mode == 0) ? addr : addr_internal;

    
    // PESOS LSTM - Separados por gate (formato 2D)
    
    
    // Forget Gate (gate 00)
    logic signed [WIDTH-1:0] lstm_wx_forget [LSTM_HIDDEN][LSTM_INPUTS];
    logic signed [WIDTH-1:0] lstm_wh_forget [LSTM_HIDDEN][LSTM_HIDDEN];
    logic signed [WIDTH-1:0] lstm_bias_forget [LSTM_HIDDEN];
    
    // Input Gate (gate 01)
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
    
    
    // PESOS ReLU
    
    
    logic signed [WIDTH-1:0] relu_w [RELU_NEURONS][RELU_INPUTS];
    logic signed [WIDTH-1:0] relu_bias [RELU_NEURONS];
    
    
    // PESOS Output Sigmoid
    
    
    logic signed [WIDTH-1:0] out_w [1][OUT_INPUTS];
    logic signed [WIDTH-1:0] out_bias [1];
    
    
    // SINAIS INTERMEDIÁRIOS
    
    
    logic signed [WIDTH-1:0] lstm_h_out [TIMESTEPS][LSTM_HIDDEN];
    logic signed [WIDTH-1:0] lstm_c_final [LSTM_HIDDEN];
    logic signed [WIDTH-1:0] relu_y [RELU_NEURONS];
    logic signed [WIDTH-1:0] out_temp [1];
    
    logic lstm_done;
    logic lstm_start;  // Nível, não pulso
    
    
    // FSM PARA CARREGAR PESOS
    
    
    typedef enum logic [5:0] {
        IDLE,
        // LSTM weights
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
        // ReLU weights
        LOAD_RELU_W, LOAD_RELU_W_DATA,
        LOAD_RELU_B, LOAD_RELU_B_DATA,
        // Output weights
        LOAD_OUT_W, LOAD_OUT_W_DATA,
        LOAD_OUT_B, LOAD_OUT_B_DATA,
        RUN_LSTM,      // Estado que mantém start ativo
        WAIT_LSTM,     // Aguarda LSTM terminar
        RUN_DONE
    } state_t;

    state_t state;
    logic [5:0] neuron;
    logic [5:0] idx;

    function logic [20:0] gen_lstm_addr(
        input logic [3:0] layer,
        input logic is_bias,
        input logic [1:0] gate,
        input logic [5:0] neuron,
        input logic recurrent,
        input logic [5:0] idx
    );
        gen_lstm_addr = {
            layer[3:0],
            is_bias,
            1'b1,
            gate[1:0],
            neuron[5:0],
            recurrent,
            idx[5:0]
        };
    endfunction

    function logic [20:0] gen_mlp_addr(
        input logic [3:0] layer,
        input logic is_bias,
        input logic [5:0] neuron,
        input logic [5:0] idx
    );
        gen_mlp_addr = {
            layer[3:0],
            is_bias,
            1'b0,
            2'b00,
            neuron[5:0],
            1'b0,
            idx[5:0]
        };
    endfunction

    // Gera sinal de start em nível (fica 1 até o LSTM terminar)
    // BLOCO SEPARADO E ÚNICO PARA lstm_start
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
            // NÃO atribuir lstm_start aqui!
        end
        else begin
            case (state)
                IDLE: begin
                    ready <= 1'b0;
                    neuron <= 0;
                    idx <= 0;
                    state <= LOAD_LSTM_WX_FORGET;
                end

                //FORGET GATE (00) =
                LOAD_LSTM_WX_FORGET: begin
                  addr_internal <= gen_lstm_addr(4'd0, 1'b0, 2'b01, neuron, 1'b0, idx);
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
                  addr_internal <= gen_lstm_addr(4'd0, 1'b0, 2'b01, neuron, 1'b1, idx);
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
                  addr_internal <= gen_lstm_addr(4'd0, 1'b1, 2'b01, neuron, 1'b0, 6'd0);
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

                //INPUT GATE (01) =
                LOAD_LSTM_WX_INPUT: begin
                  addr_internal <= gen_lstm_addr(4'd0, 1'b0, 2'b00, neuron, 1'b0, idx);
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
                  addr_internal <= gen_lstm_addr(4'd0, 1'b0, 2'b00, neuron, 1'b1, idx);
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
                  addr_internal <= gen_lstm_addr(4'd0, 1'b1, 2'b00, neuron, 1'b0, 6'd0);
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

                //CANDIDATE GATE (10) =
                LOAD_LSTM_WX_CELL: begin
                    addr_internal <= gen_lstm_addr(4'd0, 1'b0, 2'b10, neuron, 1'b0, idx);
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
                    addr_internal <= gen_lstm_addr(4'd0, 1'b0, 2'b10, neuron, 1'b1, idx);
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
                    addr_internal <= gen_lstm_addr(4'd0, 1'b1, 2'b10, neuron, 1'b0, 6'd0);
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

                //OUTPUT GATE (11) =
                LOAD_LSTM_WX_OUTPUT: begin
                    addr_internal <= gen_lstm_addr(4'd0, 1'b0, 2'b11, neuron, 1'b0, idx);
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
                    addr_internal <= gen_lstm_addr(4'd0, 1'b0, 2'b11, neuron, 1'b1, idx);
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
                    addr_internal <= gen_lstm_addr(4'd0, 1'b1, 2'b11, neuron, 1'b0, 6'd0);
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

                //ReLU WEIGHTS =
                LOAD_RELU_W: begin
                    addr_internal <= gen_mlp_addr(4'd1, 1'b0, neuron, idx);
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
                    addr_internal <= gen_mlp_addr(4'd1, 1'b1, neuron, 6'd0);
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

                //OUTPUT WEIGHTS =
                LOAD_OUT_W: begin
                    addr_internal <= gen_mlp_addr(4'd2, 1'b0, 6'd0, idx);
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
                    addr_internal <= gen_mlp_addr(4'd2, 1'b1, 6'd0, 6'd0);
                    state <= LOAD_OUT_B_DATA;
                end
                
                LOAD_OUT_B_DATA: begin
                    out_bias[0] <= data_out;
                    state <= RUN_LSTM;  // Vai para RUN_LSTM (start fica ativo)
                end

                //EXECUTA LSTM (start em nível)
                RUN_LSTM: begin
                    ready <= 1'b0;
                    if (lstm_done) begin
                        state <= WAIT_LSTM;
                    end
                end
                
                //ESPERA LSTM FINALIZAR
                WAIT_LSTM: begin
                    ready <= 1'b0;
                    if (lstm_done) begin
                        state <= RUN_DONE;
                    end
                end
                
                //EXECUÇÃO CONCLUÍDA
                RUN_DONE: begin
                    ready <= 1'b1;
                    // Fica aqui até mode mudar
                end
            endcase
        end
    end

    
    // INSTÂNCIAS DAS CAMADAS
    
    
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
        .start(lstm_start),  // Usa nível, não pulso
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
    
    // Extrai a saída do vetor de 1 elemento
    assign y_out = out_temp[0];

endmodule