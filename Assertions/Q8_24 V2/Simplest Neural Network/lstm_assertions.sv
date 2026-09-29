// LSTM VERIFICATION PLAN ASSERTIONS
// Aligned with VPLAN Rev 1.0 - Complete Feature List F1-F9
// Compatible with Cadence Xcelium

package lstm_vplan_pkg;
endpackage


module lstm_assertions
(
    input logic clk,
    input logic reset,
    input logic clear,
    input logic mode,
    input logic we,
    input logic [11:0] addr,
    input logic signed [31:0] data_in,
    input logic signed [31:0] y_out,
    input logic ready
);

    // nomes dos estados da FSM do lstm_network
    localparam logic [5:0] S_IDLE            = 6'd0;
    localparam logic [5:0] S_LOAD_LSTM_WX_FORGET      = 6'd1;
    localparam logic [5:0] S_LOAD_LSTM_WX_FORGET_DATA = 6'd2;
    localparam logic [5:0] S_LOAD_LSTM_WH_FORGET      = 6'd3;
    localparam logic [5:0] S_LOAD_LSTM_WH_FORGET_DATA = 6'd4;
    localparam logic [5:0] S_LOAD_LSTM_BIAS_FORGET      = 6'd5;
    localparam logic [5:0] S_LOAD_LSTM_BIAS_FORGET_DATA = 6'd6;
    localparam logic [5:0] S_LOAD_LSTM_WX_INPUT      = 6'd7;
    localparam logic [5:0] S_LOAD_LSTM_WX_INPUT_DATA = 6'd8;
    localparam logic [5:0] S_LOAD_LSTM_WH_INPUT      = 6'd9;
    localparam logic [5:0] S_LOAD_LSTM_WH_INPUT_DATA = 6'd10;
    localparam logic [5:0] S_LOAD_LSTM_BIAS_INPUT      = 6'd11;
    localparam logic [5:0] S_LOAD_LSTM_BIAS_INPUT_DATA = 6'd12;
    localparam logic [5:0] S_LOAD_LSTM_WX_CELL      = 6'd13;
    localparam logic [5:0] S_LOAD_LSTM_WX_CELL_DATA = 6'd14;
    localparam logic [5:0] S_LOAD_LSTM_WH_CELL      = 6'd15;
    localparam logic [5:0] S_LOAD_LSTM_WH_CELL_DATA = 6'd16;
    localparam logic [5:0] S_LOAD_LSTM_BIAS_CELL      = 6'd17;
    localparam logic [5:0] S_LOAD_LSTM_BIAS_CELL_DATA = 6'd18;
    localparam logic [5:0] S_LOAD_LSTM_WX_OUTPUT      = 6'd19;
    localparam logic [5:0] S_LOAD_LSTM_WX_OUTPUT_DATA = 6'd20;
    localparam logic [5:0] S_LOAD_LSTM_WH_OUTPUT      = 6'd21;
    localparam logic [5:0] S_LOAD_LSTM_WH_OUTPUT_DATA = 6'd22;
    localparam logic [5:0] S_LOAD_LSTM_BIAS_OUTPUT      = 6'd23;
    localparam logic [5:0] S_LOAD_LSTM_BIAS_OUTPUT_DATA = 6'd24;
    localparam logic [5:0] S_LOAD_RELU_W      = 6'd25;
    localparam logic [5:0] S_LOAD_RELU_W_DATA = 6'd26;
    localparam logic [5:0] S_LOAD_RELU_B      = 6'd27;
    localparam logic [5:0] S_LOAD_RELU_B_DATA = 6'd28;
    localparam logic [5:0] S_LOAD_OUT_W      = 6'd29;
    localparam logic [5:0] S_LOAD_OUT_W_DATA = 6'd30;
    localparam logic [5:0] S_LOAD_OUT_B      = 6'd31;
    localparam logic [5:0] S_LOAD_OUT_B_DATA = 6'd32;
    localparam logic [5:0] S_RUN_LSTM  = 6'd33;
    localparam logic [5:0] S_WAIT_LSTM = 6'd34;
    localparam logic [5:0] S_RUN_DONE  = 6'd35;

    // 1) RESET

    property p_reset_ready_low;
        @(posedge clk) reset |=> (ready == 1'b0);
    endproperty
    a_reset_ready_low: assert property (p_reset_ready_low)
        else $error("[ASSERT] ready != 0 apos reset");

    property p_reset_yout_zero;
        @(posedge clk) reset |=> (y_out == 32'sd0);
    endproperty
    a_reset_yout_zero: assert property (p_reset_yout_zero)
        else $error("[ASSERT] y_out != 0 apos reset");

    property p_reset_clears_ready;
        @(posedge clk) reset |-> ##1 (ready == 1'b0);
    endproperty
    a_reset_clears_ready: assert property (p_reset_clears_ready)
        else $error("[ASSERT] reset nao limpou ready em 1 ciclo");

    // 2) CLEAR

    property p_clear_yout_zero;
        @(posedge clk) clear |-> (y_out == 32'sd0);
    endproperty
    a_clear_yout_zero: assert property (p_clear_yout_zero)
        else $error("[ASSERT] y_out != 0 durante clear");

    property p_clear_no_ready;
        @(posedge clk) clear |=> (ready == 1'b0);
    endproperty
    a_clear_no_ready: assert property (p_clear_no_ready)
        else $error("[ASSERT] clear disparou ready indevidamente");

    // 3) SEQUENCIA DE READY

    property p_ready_only_in_exec;
        @(posedge clk) disable iff (reset)
        (ready == 1'b1) |-> (mode == 1'b1);
    endproperty
    a_ready_only_in_exec: assert property (p_ready_only_in_exec)
        else $error("[ASSERT] ready=1 fora do modo de execucao");

    localparam int READY_MAX_CYCLES = 1000;
    property p_ready_bounded;
        @(posedge clk) disable iff (reset)
        $rose(ready) |-> ##[1:READY_MAX_CYCLES] $fell(ready);
    endproperty
    a_ready_bounded: assert property (p_ready_bounded)
        else $error("[ASSERT] ready ficou alto por mais de %0d ciclos", READY_MAX_CYCLES);

    property p_ready_not_in_write;
        @(posedge clk) disable iff (reset)
        (mode == 1'b0) |-> (ready == 1'b0);
    endproperty
    a_ready_not_in_write: assert property (p_ready_not_in_write)
        else $error("[ASSERT] ready=1 durante mode=0 (escrita)");

    // 4) MODOS DE OPERACAO

    property p_no_write_in_exec;
        @(posedge clk) disable iff (reset)
        (mode == 1'b1) |-> (we == 1'b0);
    endproperty
    a_no_write_in_exec: assert property (p_no_write_in_exec)
        else $error("[ASSERT] we=1 durante mode=1 (execucao)");

    // 5) VALIDADE DA SAIDA

    property p_yout_range;
        @(posedge clk) disable iff (reset)
        (ready == 1'b1) |-> (y_out >= 32'sd0 && y_out <= 32'sh01000000);
    endproperty
    a_yout_range: assert property (p_yout_range)
        else $error("[ASSERT] y_out fora de [0,1] em Q8.24: %0d", y_out);

    // 6) OVERFLOW / SATURACAO
    // property p_no_overflow;
    //     @(posedge clk) disable iff (reset) !dut.overflow;
    // endproperty
    // a_no_overflow: assert property (p_no_overflow)
    //     else $error("[ASSERT] overflow detectado");

    // 7) ESTADOS DA FSM (lstm_network)

    property p_fsm_valid_state;
        @(posedge clk) disable iff (reset)
        (dut.state inside {
            S_IDLE,
            S_LOAD_LSTM_WX_FORGET, S_LOAD_LSTM_WX_FORGET_DATA,
            S_LOAD_LSTM_WH_FORGET, S_LOAD_LSTM_WH_FORGET_DATA,
            S_LOAD_LSTM_BIAS_FORGET, S_LOAD_LSTM_BIAS_FORGET_DATA,
            S_LOAD_LSTM_WX_INPUT, S_LOAD_LSTM_WX_INPUT_DATA,
            S_LOAD_LSTM_WH_INPUT, S_LOAD_LSTM_WH_INPUT_DATA,
            S_LOAD_LSTM_BIAS_INPUT, S_LOAD_LSTM_BIAS_INPUT_DATA,
            S_LOAD_LSTM_WX_CELL, S_LOAD_LSTM_WX_CELL_DATA,
            S_LOAD_LSTM_WH_CELL, S_LOAD_LSTM_WH_CELL_DATA,
            S_LOAD_LSTM_BIAS_CELL, S_LOAD_LSTM_BIAS_CELL_DATA,
            S_LOAD_LSTM_WX_OUTPUT, S_LOAD_LSTM_WX_OUTPUT_DATA,
            S_LOAD_LSTM_WH_OUTPUT, S_LOAD_LSTM_WH_OUTPUT_DATA,
            S_LOAD_LSTM_BIAS_OUTPUT, S_LOAD_LSTM_BIAS_OUTPUT_DATA,
            S_LOAD_RELU_W, S_LOAD_RELU_W_DATA,
            S_LOAD_RELU_B, S_LOAD_RELU_B_DATA,
            S_LOAD_OUT_W, S_LOAD_OUT_W_DATA,
            S_LOAD_OUT_B, S_LOAD_OUT_B_DATA,
            S_RUN_LSTM, S_WAIT_LSTM, S_RUN_DONE
        });
    endproperty
    a_fsm_valid_state: assert property (p_fsm_valid_state)
        else $error("[ASSERT] estado invalido da FSM: %0d", dut.state);

    property p_fsm_no_idle_to_done;
        @(posedge clk) disable iff (reset)
        (dut.state == S_IDLE) |=> (dut.state != S_RUN_DONE);
    endproperty
    a_fsm_no_idle_to_done: assert property (p_fsm_no_idle_to_done)
        else $error("[ASSERT] FSM pulou IDLE -> RUN_DONE");

    // 8) ESTADOS DA FSM (lstm_layer)

    // estados reais do lstm_layer: IDLE, COMPUTE, DONE
    property p_lstm_valid_state;
        @(posedge clk) disable iff (reset)
        (dut.lstm_inst.state inside {2'd0, 2'd1, 2'd2});
    endproperty
    a_lstm_valid_state: assert property (p_lstm_valid_state)
        else $error("[ASSERT] estado invalido da FSM da LSTM: %0d", dut.lstm_inst.state);

    // 9) CONSISTENCIA DOS DADOS
    // property p_yout_stable_when_invalid;
    //     @(posedge clk) disable iff (reset)
    //     (!dut.valid_data) |=> $stable(y_out);
    // endproperty
    // a_yout_stable_when_invalid: assert property (p_yout_stable_when_invalid)
    //     else $error("[ASSERT] y_out mudou sem valid_data");

    // 10) COBERTURA

    covergroup cg_modes @(posedge clk);
        cp_mode: coverpoint mode {
            bins escrita  = {1'b0};
            bins execucao = {1'b1};
        }
    endgroup

    covergroup cg_reset @(posedge clk);
        cp_reset: coverpoint reset {
            bins idle  = {1'b0};
            bins ativo = {1'b1};
        }
    endgroup

    covergroup cg_ready @(posedge clk);
        cp_ready: coverpoint ready {
            bins low  = {1'b0};
            bins high = {1'b1};
        }
    endgroup

    covergroup cg_yout @(posedge clk);
        cp_yout: coverpoint y_out {
            bins prox_zero     = {[32'sd0        : 32'sd4194304]};
            bins intermediario = {[32'sd4194305  : 32'sd12582912]};
            bins prox_um       = {[32'sd12582913 : 32'sh01000000]};
        }
    endgroup

    covergroup cg_lstm_state @(posedge clk);
        cp_lstm_state: coverpoint dut.lstm_inst.state {
            bins idle    = {2'd0};
            bins compute = {2'd1};
            bins done    = {2'd2};
        }
    endgroup

    cg_modes cg_modes_inst = new;
    cg_reset cg_reset_inst = new;
    cg_ready cg_ready_inst = new;
    cg_yout  cg_yout_inst  = new;
    cg_lstm_state cg_lstm_state_inst = new;

endmodule


bind lstm_network lstm_assertions u_assertions (
    .clk     (clk),
    .reset   (reset),
    .clear   (clear),
    .mode    (mode),
    .we      (we),
    .addr    (addr),
    .data_in (data_in),
    .y_out   (y_out),
    .ready   (ready)
);