module weight_bank #(
    parameter int WIDTH = 32,
    parameter int N_LAYERS = 3,
    parameter int LAYER_SIZES [0:N_LAYERS-1] = '{2,100,1},
    parameter int MAX_SIZE = 2048
)(
    input  logic clk,
    input  logic we,
    input  logic [10:0] addr,
    input  logic signed [WIDTH-1:0] data_in,

    output logic signed [WIDTH-1:0] data_out
);

    logic signed [WIDTH-1:0] mem [0:MAX_SIZE-1];

    // decode addr
    logic [1:0] layer;
    logic       is_bias;
    logic [3:0] neuron;
    logic [3:0] idx;

    assign layer   = addr[10:9];
    assign is_bias = addr[8];
    assign neuron  = addr[7:4];
    assign idx     = addr[3:0];

    // escrita
    always_ff @(posedge clk) begin
        if (we)
            mem[addr] <= data_in;
    end

    // leitura
    assign data_out = mem[addr];

endmodule