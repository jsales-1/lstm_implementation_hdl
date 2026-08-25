module sigmoid #(
    parameter int WIDTH = 32,
    parameter int FRAC  = 24
)(
    input  logic signed [WIDTH-1:0] z,
    output logic signed [WIDTH-1:0] y
);

    logic signed [2*WIDTH-1:0] mult;
    logic signed [WIDTH-1:0] z2, z3, z5, z7, z9;

    localparam signed [WIDTH-1:0] HALF   = 32'sd8388608;   // 0.5
    localparam signed [WIDTH-1:0] ONE    = 32'sd16777216;  // 1.0
    localparam signed [WIDTH-1:0] ZERO   = 32'sd0;
    localparam signed [WIDTH-1:0] SAT_P  = 32'sd50331648;  // +3.0
    localparam signed [WIDTH-1:0] SAT_N  = -32'sd50331648; // -3.0

    always_comb begin
        if (z <= SAT_N) begin
            y = ZERO;
        end
        else if (z >= SAT_P) begin
            y = ONE;
        end
        else begin
            mult = z * z;
            z2   = mult >>> FRAC;
            mult = z2 * z;
            z3   = mult >>> FRAC;
            mult = z3 * z2;
            z5   = mult >>> FRAC;
            mult = z5 * z2;
            z7   = mult >>> FRAC;
            mult = z7 * z2;
            z9   = mult >>> FRAC;

            y = HALF + (z >>> 2) - (z3 / 48) + (z5 / 480) - ((17 * z7) / 80640) + ((31 * z9) / 1451520);

            if (y < ZERO) y = ZERO;
            else if (y > ONE) y = ONE;
        end
    end

endmodule


module tanh #(
    parameter int WIDTH = 32,
    parameter int FRAC  = 24
)(
    input  logic signed [WIDTH-1:0] z,
    output logic signed [WIDTH-1:0] y
);

    logic signed [2*WIDTH-1:0] mult;
    logic signed [WIDTH-1:0] z2, z3, z5, z7, z9;

    localparam signed [WIDTH-1:0] ONE     = 32'sd16777216;  // 1.0
    localparam signed [WIDTH-1:0] NEG_ONE = -32'sd16777216; // -1.0
    localparam signed [WIDTH-1:0] SAT_P   = 32'sd50331648;  // +3.0
    localparam signed [WIDTH-1:0] SAT_N   = -32'sd50331648; // -3.0

    always_comb begin
        if (z <= SAT_N) begin
            y = NEG_ONE;
        end
        else if (z >= SAT_P) begin
            y = ONE;
        end
        else begin
            mult = z * z;
            z2   = mult >>> FRAC;
            mult = z2 * z;
            z3   = mult >>> FRAC;
            mult = z3 * z2;
            z5   = mult >>> FRAC;
            mult = z5 * z2;
            z7   = mult >>> FRAC;
            mult = z7 * z2;
            z9   = mult >>> FRAC;

            y = z - (z3 / 3) + ((2 * z5) / 15) - ((17 * z7) / 315) + ((62 * z9) / 2835);

            if (y < NEG_ONE) y = NEG_ONE;
            else if (y > ONE) y = ONE;
        end
    end

endmodule


module mac #(
    parameter int N     = 4,
    parameter int WIDTH = 32,
    parameter int FRAC  = 24
)(
    input  logic signed [WIDTH-1:0] x [N],
    input  logic signed [WIDTH-1:0] w [N],
    input  logic signed [WIDTH-1:0] bias,
    output logic signed [WIDTH-1:0] sum
);

    integer i;
    logic signed [WIDTH-1:0] acc;
    logic signed [2*WIDTH-1:0] mult;

    always_comb begin
        acc = bias;
        for (i = 0; i < N; i++) begin
            mult = x[i] * w[i];
            acc = acc + (mult >>> FRAC);
        end
        sum = acc;
    end

endmodule


module lstm_cell_neuron #(
    parameter int N_INPUTS  = 4,
    parameter int N_HIDDEN  = 4,
    parameter int WIDTH     = 32,
    parameter int FRAC      = 24
)(
    input  logic signed [WIDTH-1:0] x [N_INPUTS],
    input  logic signed [WIDTH-1:0] h_prev [N_HIDDEN],
    input  logic signed [WIDTH-1:0] c_prev,
    
    input  logic signed [WIDTH-1:0] w_ix [N_INPUTS],
    input  logic signed [WIDTH-1:0] w_ih [N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_fx [N_INPUTS],
    input  logic signed [WIDTH-1:0] w_fh [N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_ox [N_INPUTS],
    input  logic signed [WIDTH-1:0] w_oh [N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_cx [N_INPUTS],
    input  logic signed [WIDTH-1:0] w_ch [N_HIDDEN],
    
    input  logic signed [WIDTH-1:0] bias_i,
    input  logic signed [WIDTH-1:0] bias_f,
    input  logic signed [WIDTH-1:0] bias_o,
    input  logic signed [WIDTH-1:0] bias_c,
    
    output logic signed [WIDTH-1:0] h_out,
    output logic signed [WIDTH-1:0] c_out
);

    logic signed [WIDTH-1:0] concat_xh [N_INPUTS + N_HIDDEN];
    logic signed [WIDTH-1:0] w_i_concat [N_INPUTS + N_HIDDEN];
    logic signed [WIDTH-1:0] w_f_concat [N_INPUTS + N_HIDDEN];
    logic signed [WIDTH-1:0] w_o_concat [N_INPUTS + N_HIDDEN];
    logic signed [WIDTH-1:0] w_c_concat [N_INPUTS + N_HIDDEN];
    
    logic signed [WIDTH-1:0] sum_i, sum_f, sum_o, sum_c;
    logic signed [WIDTH-1:0] i_t, f_t, o_t, c_tilde;
    logic signed [WIDTH-1:0] tanh_c_out;
    logic signed [WIDTH-1:0] c_out_temp;
    
    logic signed [2*WIDTH-1:0] mult_f, mult_i, mult_o;
    
    integer j;
    
    always_comb begin
        for (j = 0; j < N_INPUTS; j++) begin
            concat_xh[j] = x[j];
        end
        for (j = 0; j < N_HIDDEN; j++) begin
            concat_xh[N_INPUTS + j] = h_prev[j];
        end
        
        for (j = 0; j < N_INPUTS; j++) begin
            w_i_concat[j] = w_ix[j];
            w_f_concat[j] = w_fx[j];
            w_o_concat[j] = w_ox[j];
            w_c_concat[j] = w_cx[j];
        end
        for (j = 0; j < N_HIDDEN; j++) begin
            w_i_concat[N_INPUTS + j] = w_ih[j];
            w_f_concat[N_INPUTS + j] = w_fh[j];
            w_o_concat[N_INPUTS + j] = w_oh[j];
            w_c_concat[N_INPUTS + j] = w_ch[j];
        end
    end
    
    mac #(.N(N_INPUTS + N_HIDDEN), .WIDTH(WIDTH), .FRAC(FRAC)) mac_i (
        .x(concat_xh), .w(w_i_concat), .bias(bias_i), .sum(sum_i)
    );
    
    mac #(.N(N_INPUTS + N_HIDDEN), .WIDTH(WIDTH), .FRAC(FRAC)) mac_f (
        .x(concat_xh), .w(w_f_concat), .bias(bias_f), .sum(sum_f)
    );
    
    mac #(.N(N_INPUTS + N_HIDDEN), .WIDTH(WIDTH), .FRAC(FRAC)) mac_o (
        .x(concat_xh), .w(w_o_concat), .bias(bias_o), .sum(sum_o)
    );
    
    mac #(.N(N_INPUTS + N_HIDDEN), .WIDTH(WIDTH), .FRAC(FRAC)) mac_c (
        .x(concat_xh), .w(w_c_concat), .bias(bias_c), .sum(sum_c)
    );
    
    sigmoid #(.WIDTH(WIDTH), .FRAC(FRAC)) sig_i (.z(sum_i), .y(i_t));
    sigmoid #(.WIDTH(WIDTH), .FRAC(FRAC)) sig_f (.z(sum_f), .y(f_t));
    sigmoid #(.WIDTH(WIDTH), .FRAC(FRAC)) sig_o (.z(sum_o), .y(o_t));
    tanh   #(.WIDTH(WIDTH), .FRAC(FRAC)) tanh_c (.z(sum_c), .y(c_tilde));
    
    tanh #(.WIDTH(WIDTH), .FRAC(FRAC)) tanh_cell (.z(c_out_temp), .y(tanh_c_out));
    
    always_comb begin
        mult_f = f_t * c_prev;
        mult_i = i_t * c_tilde;
        c_out_temp = (mult_f >>> FRAC) + (mult_i >>> FRAC);
        c_out = c_out_temp;
        
        mult_o = o_t * tanh_c_out;
        h_out = mult_o >>> FRAC;
    end

endmodule


module lstm_cell #(
    parameter int N_INPUTS  = 4,
    parameter int N_HIDDEN  = 4,
    parameter int WIDTH     = 32,
    parameter int FRAC      = 24
)(
    input  logic signed [WIDTH-1:0] x [N_INPUTS],
    input  logic signed [WIDTH-1:0] h_prev [N_HIDDEN],
    input  logic signed [WIDTH-1:0] c_prev [N_HIDDEN],
    
    input  logic signed [WIDTH-1:0] w_ix [N_HIDDEN][N_INPUTS],
    input  logic signed [WIDTH-1:0] w_ih [N_HIDDEN][N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_fx [N_HIDDEN][N_INPUTS],
    input  logic signed [WIDTH-1:0] w_fh [N_HIDDEN][N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_ox [N_HIDDEN][N_INPUTS],
    input  logic signed [WIDTH-1:0] w_oh [N_HIDDEN][N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_cx [N_HIDDEN][N_INPUTS],
    input  logic signed [WIDTH-1:0] w_ch [N_HIDDEN][N_HIDDEN],
    
    input  logic signed [WIDTH-1:0] bias_i [N_HIDDEN],
    input  logic signed [WIDTH-1:0] bias_f [N_HIDDEN],
    input  logic signed [WIDTH-1:0] bias_o [N_HIDDEN],
    input  logic signed [WIDTH-1:0] bias_c [N_HIDDEN],
    
    output logic signed [WIDTH-1:0] h_out [N_HIDDEN],
    output logic signed [WIDTH-1:0] c_out [N_HIDDEN]
);

    genvar n;
    generate
        for (n = 0; n < N_HIDDEN; n++) begin : GEN_LSTM_NEURONS
            lstm_cell_neuron #(
                .N_INPUTS(N_INPUTS),
                .N_HIDDEN(N_HIDDEN),
                .WIDTH(WIDTH),
                .FRAC(FRAC)
            ) neuron (
                .x(x),
                .h_prev(h_prev),
                .c_prev(c_prev[n]),
                .w_ix(w_ix[n]),
                .w_ih(w_ih[n]),
                .w_fx(w_fx[n]),
                .w_fh(w_fh[n]),
                .w_ox(w_ox[n]),
                .w_oh(w_oh[n]),
                .w_cx(w_cx[n]),
                .w_ch(w_ch[n]),
                .bias_i(bias_i[n]),
                .bias_f(bias_f[n]),
                .bias_o(bias_o[n]),
                .bias_c(bias_c[n]),
                .h_out(h_out[n]),
                .c_out(c_out[n])
            );
        end
    endgenerate

endmodule


module lstm_layer #(
    parameter int N_INPUTS   = 4,
    parameter int N_HIDDEN   = 4,
    parameter int N_TIMESTEPS = 4,
    parameter int WIDTH      = 32,
    parameter int FRAC       = 24
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    
    input  logic signed [WIDTH-1:0] x [N_TIMESTEPS][N_INPUTS],
    
    input  logic signed [WIDTH-1:0] w_ix [N_HIDDEN][N_INPUTS],
    input  logic signed [WIDTH-1:0] w_ih [N_HIDDEN][N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_fx [N_HIDDEN][N_INPUTS],
    input  logic signed [WIDTH-1:0] w_fh [N_HIDDEN][N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_ox [N_HIDDEN][N_INPUTS],
    input  logic signed [WIDTH-1:0] w_oh [N_HIDDEN][N_HIDDEN],
    input  logic signed [WIDTH-1:0] w_cx [N_HIDDEN][N_INPUTS],
    input  logic signed [WIDTH-1:0] w_ch [N_HIDDEN][N_HIDDEN],
    
    input  logic signed [WIDTH-1:0] bias_i [N_HIDDEN],
    input  logic signed [WIDTH-1:0] bias_f [N_HIDDEN],
    input  logic signed [WIDTH-1:0] bias_o [N_HIDDEN],
    input  logic signed [WIDTH-1:0] bias_c [N_HIDDEN],
    
    output logic signed [WIDTH-1:0] h_out [N_TIMESTEPS][N_HIDDEN],
    output logic signed [WIDTH-1:0] c_final [N_HIDDEN],
    output logic done
);

    logic signed [WIDTH-1:0] h_curr [N_HIDDEN];
    logic signed [WIDTH-1:0] c_curr [N_HIDDEN];
    logic signed [WIDTH-1:0] h_next [N_HIDDEN];
    logic signed [WIDTH-1:0] c_next [N_HIDDEN];
    
    logic [31:0] timestep;
    
    typedef enum {IDLE, COMPUTE, DONE} state_t;
    state_t state;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            timestep <= 0;
            done <= 0;
            for (int i = 0; i < N_HIDDEN; i++) begin
                h_curr[i] <= 0;
                c_curr[i] <= 0;
            end
        end
        else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= COMPUTE;
                        timestep <= 0;
                        for (int i = 0; i < N_HIDDEN; i++) begin
                            h_curr[i] <= 0;
                            c_curr[i] <= 0;
                        end
                    end
                end
                
                COMPUTE: begin
                    if (timestep < N_TIMESTEPS) begin
                        h_curr <= h_next;
                        c_curr <= c_next;
                        timestep <= timestep + 1;
                        
                        // CORREÇÃO: Atribui elemento por elemento
                        for (int i = 0; i < N_HIDDEN; i++) begin
                            h_out[timestep][i] <= h_next[i];
                        end
                        
                        if (timestep == N_TIMESTEPS - 1) begin
                            for (int i = 0; i < N_HIDDEN; i++) begin
                                c_final[i] <= c_next[i];
                            end
                            state <= DONE;
                        end
                    end
                end
                
                DONE: begin
                    done <= 1;
                    if (!start) begin
                        state <= IDLE;
                    end
                end
            endcase
        end
    end
    
    lstm_cell #(
        .N_INPUTS(N_INPUTS),
        .N_HIDDEN(N_HIDDEN),
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) cell_inst (
        .x(x[timestep]),
        .h_prev(h_curr),
        .c_prev(c_curr),
        .w_ix(w_ix),
        .w_ih(w_ih),
        .w_fx(w_fx),
        .w_fh(w_fh),
        .w_ox(w_ox),
        .w_oh(w_oh),
        .w_cx(w_cx),
        .w_ch(w_ch),
        .bias_i(bias_i),
        .bias_f(bias_f),
        .bias_o(bias_o),
        .bias_c(bias_c),
        .h_out(h_next),
        .c_out(c_next)
    );

endmodule