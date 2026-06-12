module weight_bank #(
    parameter int WIDTH = 32,
    parameter int MAX_SIZE = 2097152
)(
    input  logic clk,
    input  logic we,

    input  logic [31:0] addr,
    input  logic signed [WIDTH-1:0] data_in,

    output logic signed [WIDTH-1:0] data_out
);

    // ============================================================
    // MEMORY
    // ============================================================

    logic signed [WIDTH-1:0] mem [0:MAX_SIZE-1];

    // ============================================================
    // ADDRESS FORMAT
    // ============================================================
    //
    // [31:24] -> layer
    // [23]    -> is_bias
    // [22]    -> is_lstm
    // [21:20] -> gate
    // [19:10] -> neuron
    // [9]     -> recurrent
    // [8:0]   -> idx
    //
    // gate:
    // 00 -> forget
    // 01 -> input
    // 10 -> candidate
    // 11 -> output
    //
    // recurrent:
    // 0 -> Wx
    // 1 -> Wh
    //
    // ============================================================

    logic [7:0] layer;
    logic       is_bias;
    logic       is_lstm;
    logic [1:0] gate;
    logic [9:0] neuron;
    logic       recurrent;
    logic [8:0] idx;

    assign layer      = addr[31:24];
    assign is_bias    = addr[23];
    assign is_lstm    = addr[22];
    assign gate       = addr[21:20];
    assign neuron     = addr[19:10];
    assign recurrent  = addr[9];
    assign idx        = addr[8:0];

    // ============================================================
    // WRITE
    // ============================================================

    always_ff @(posedge clk) begin
        if (we)
            mem[addr] <= data_in;
    end

    // ============================================================
    // READ
    // ============================================================

    assign data_out = mem[addr];

endmodule