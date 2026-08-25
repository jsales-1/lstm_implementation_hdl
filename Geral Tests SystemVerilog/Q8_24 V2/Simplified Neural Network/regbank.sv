module weight_bank #(
    parameter int WIDTH = 32,
    parameter int MAX_SIZE = 4096  // 2^12 = 4096
)(
    input  logic clk,
    input  logic rst,
    input  logic we,

    input  logic [11:0] addr,     
    input  logic signed [WIDTH-1:0] data_in,

    output logic signed [WIDTH-1:0] data_out
);

    logic signed [WIDTH-1:0] mem [0:MAX_SIZE-1];

    // ============================================================
    // DECODIFICAÇÃO (12 bits)
    // ============================================================
    // Bits 11-10: layer     (2 bits) → 0 a 3
    // Bit  9:     is_bias   (1 bit)  → 0 = peso, 1 = bias
    // Bits 8-7:   gate      (2 bits) → 0 a 3
    // Bits 6-4:   neuron    (3 bits) → 0 a 7
    // Bit  3:     recurrent (1 bit)  → 0 = Wx, 1 = Wh
    // Bits 2-0:   idx       (3 bits) → 0 a 7

    logic [1:0]  layer;
    logic        is_bias;
    logic [1:0]  gate;
    logic [2:0]  neuron;
    logic        recurrent;
    logic [2:0]  idx;

    assign layer     = addr[11:10];
    assign is_bias   = addr[9];
    assign gate      = addr[8:7];
    assign neuron    = addr[6:4];
    assign recurrent = addr[3];
    assign idx       = addr[2:0];

    // MAPEAMENTO DOS GATES (MANTIDO O QUE FUNCIONA)
    // gate:
    // 00 -> input  
    // 01 -> forget 
    // 10 -> candidate
    // 11 -> output
    //
    // recurrent:
    // 0 -> Wx
    // 1 -> Wh

    always_ff @(posedge clk) begin
        if (rst) begin 
            integer i;
            for (i = 0; i < MAX_SIZE; i++)
                mem[i] = 0;
        end
        if (we)
            mem[addr] <= data_in;
    end

    assign data_out = mem[addr];

endmodule