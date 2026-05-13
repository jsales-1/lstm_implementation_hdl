`timescale 1ns/1ps

module tb_small_network;

    parameter int WIDTH = 32;

    logic clk;

    initial clk = 0;

    always #5 clk = ~clk;

    
    // Interface DUT
    

    logic mode; // 0 = WRITE | 1 = RUN
    logic we;
    logic [10:0] addr;
    logic signed [WIDTH-1:0] data_in;
    logic signed [WIDTH-1:0] x [2];
    logic signed [WIDTH-1:0] y [1];
  	shortreal y_frac;
    logic ready;

    
    // dut
    

    small_network dut (
        .clk(clk),
        .mode(mode),
        .we(we),
        .addr(addr),
        .data_in(data_in),
        .x1(x),
        .y2(y),
        .ready(ready)
    );

    
    //Memória Local
    

    logic signed [31:0] mem [0:2047];

    initial begin
        $readmemh("weights.mem", mem);
    end

    
    //Controle
    

    integer i;

    initial begin
    $dumpfile("wave.vcd");
    $dumpvars(0, tb_small_network);

        
        // Init
        

        mode    = 0;
        we      = 0;

        addr    = 0;
        data_in = 0;

        x[0]    = 0;
        x[1]    = 0;

        
        //Espera inicial
        #20;
  
        //Mode de Escrita
        

        $display("=");
        $display("Escrevendo pesos e bias na memória local");
        $display("=");

        for (i = 0; i < 2048; i++) begin

            @(posedge clk);

            we      <= 1'b1;
            addr    <= i[10:0];
            data_in <= mem[i];

        end

        @(posedge clk);

        we <= 1'b0;
        
        //Input de teste

        x[0] = 32'sd65536; // 1.0
        x[1] = 32'sd32768; // 0.5

        
        //Mode de Execução
        

        @(posedge clk);

        mode <= 1'b1;

        
        //Espera do ready
        wait (ready == 1'b1);
      
        y_frac = y[0]/65536.0;

        //Progação final
        #20;

        
        //Resultados
        $display("Resultado");

        $display("x0 = %d", x[0]);
        $display("x1 = %d", x[1]);

        $display("y  = %d", y[0]);
        $display("yd  = %.4f", y_frac);
  

        $finish;

    end

endmodule