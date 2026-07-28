module weight_bank #(
    parameter int WIDTH = 32,
    parameter int MAX_SIZE = 2097152
)(
    input  logic clk,
    input  logic we,

    input  logic [20:0] addr,
    input  logic signed [WIDTH-1:0] data_in,

    output logic signed [WIDTH-1:0] data_out
);

    logic signed [WIDTH-1:0] mem [0:MAX_SIZE-1];

    logic [3:0]  layer;
    logic        is_bias;
    logic        is_lstm;
    logic [1:0]  gate;
    logic [5:0]  neuron;
    logic        recurrent;
    logic [5:0]  idx;

    assign layer     = addr[20:17];
    assign is_bias   = addr[16];
    assign is_lstm   = addr[15];
    assign gate      = addr[14:13];
    assign neuron    = addr[12:7];
    assign recurrent = addr[6];
    assign idx       = addr[5:0];

  	// gate:
    // 00 -> forget
    // 01 -> input
    // 10 -> candidate
    // 11 -> output
    //
    // recurrent:
    // 0 -> Wx
    // 1 -> Wh
    initial begin
        integer i;
        for (i = 0; i < MAX_SIZE; i++)
            mem[i] = 0;
    end

    always_ff @(posedge clk) begin
        if (we)
            mem[addr] <= data_in;
    end

    assign data_out = mem[addr];

endmodule