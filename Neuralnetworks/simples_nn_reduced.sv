
module small_network #(
    parameter int WIDTH = 32,
    parameter int FRAC  = 16,

    parameter int L1_INPUTS  = 2,
    parameter int L1_NEURONS = 4,

    parameter int L2_INPUTS  = 4,
    parameter int L2_NEURONS = 1 
    )(
    input  logic clk,
    input  logic mode, // 0 = WRITE | 1 = RUN

    // escrita memória
    input  logic we,
    input  logic [10:0] addr,
    input  logic signed [WIDTH-1:0] data_in,

    // entradas
    input  logic signed [WIDTH-1:0] x1 [L1_INPUTS],

    // saída
    output logic signed [WIDTH-1:0] y2 [L2_NEURONS],

    output logic ready
);
    
    logic [10:0] addr_mux;
    logic [10:0] addr_internal;

    logic signed [WIDTH-1:0] data_out;

    weight_bank wb (
        .clk(clk),
        .we(mode ? 1'b0 : we),
        .addr(addr_mux),
        .data_in(data_in),
        .data_out(data_out)
    );

    assign addr_mux = (mode == 0) ? addr : addr_internal;

    // Pesos e vieses    
    logic signed [WIDTH-1:0] w1 [L1_NEURONS][L1_INPUTS];
    logic signed [WIDTH-1:0] b1 [L1_NEURONS];

    logic signed [WIDTH-1:0] w2 [L2_NEURONS][L2_INPUTS];
    logic signed [WIDTH-1:0] b2 [L2_NEURONS];

    logic signed [WIDTH-1:0] y1 [L1_NEURONS];

    // FSM
    
    typedef enum logic [3:0] {

        IDLE,

        LOA_DL1_W_ADDR,
        LOAD_L1_W_DATA,

        LOAD_L1_B_ADDR,
        LOAD_L1_B_DATA,

        LOAD_L2_W_ADDR,
        LOAD_L2_W_DATA,

        LOAD_L2_B_ADDR,
        LOAD_L2_B_DATA,

        RUN

    } state_t;

    state_t state;

    integer n;
    integer i;

    
    // FSM
    
    always_ff @(posedge clk or posedge reset) begin
       

        if (mode == 0) begin

            state <= IDLE;
            ready <= 0;

        end
 
        // Run Mode
        else begin

            case (state)
                

                IDLE: begin

                    n <= 0;
                    i <= 0;

                    ready <= 0;

                    state <= LOAD_L1_W_ADDR;
                end


                LOAD_L1_W_ADDR: begin

                    addr_internal <= {
                        2'd0,
                        1'b0,
                        n[3:0],
                        i[3:0]
                    };

                    state <= LOAD_L1_W_DATA;

                end


                LOAD_L1_W_DATA: begin

                    w1[n][i] <= data_out;

                    if (i == L1_INPUTS-1) begin

                        i <= 0;

                        if (n == L1_NEURONS-1) begin

                            n <= 0;
                            state <= LOAD_L1_B_ADDR;
                        end
                        else begin

                            n <= n + 1;
                            state <= LOAD_L1_W_ADDR;
                        end
                    end
                    else begin

                        i <= i + 1;
                        state <= LOAD_L1_W_ADDR;
                    end
                end


                LOAD_L1_B_ADDR: begin

                    addr_internal <= {
                        2'd0,
                        1'b1,
                        n[3:0],
                        4'd0
                    };

                    state <= LOAD_L1_B_DATA;
                end


                LOAD_L1_B_DATA: begin

                    b1[n] <= data_out;

                    if (n == L1_NEURONS-1) begin

                        n <= 0;
                        state <= LOAD_L2_W_ADDR;
                    end
                    else begin

                        n <= n + 1;
                        state <= LOAD_L1_B_ADDR;
                    end

                end


                LOAD_L2_W_ADDR: begin

                    addr_internal <= {
                        2'd1,
                        1'b0,
                        4'd0,
                        i[3:0]
                    };

                    state <= LOAD_L2_W_DATA;

                end

                
                LOAD_L2_W_DATA: begin

                    w2[0][i] <= data_out;

                    if (i == L2_INPUTS-1) begin

                        i <= 0;
                        state <= LOAD_L2_B_ADDR;

                    end
                    else begin

                        i <= i + 1;
                        state <= LOAD_L2_W_ADDR;

                    end

                end


                LOAD_L2_B_ADDR: begin

                    addr_internal <= {
                        2'd1,
                        1'b1,
                        4'd0,
                        4'd0
                    };

                    state <= LOAD_L2_B_DATA;

                end

                 

                LOAD_L2_B_DATA: begin

                    b2[0] <= data_out;

                    state <= RUN;

                end

                
                RUN: begin

                    ready <= 1'b1;

                end

            endcase
        end
    end

        // Camada 1
    
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

        // Camada 2
    
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