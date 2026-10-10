`timescale 1ns/1ps

module tb_lstm_network_netlist;

    parameter WIDTH = 32;
    parameter FRAC  = 24;
    
    parameter LSTM_INPUTS  = 4;
    parameter LSTM_HIDDEN  = 2;
    parameter TIMESTEPS    = 5;
    parameter RELU_INPUTS  = 2;
    parameter RELU_NEURONS = 4;
    parameter OUT_INPUTS   = 4;
    
    parameter NUM_FILES = 10000;
    parameter real THRESHOLD = 0.5;
    
    reg clk;
    reg reset;
    reg clear;
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    reg mode;
    reg we;
    reg [11:0] addr;
    reg signed [WIDTH-1:0] data_in;
    wire signed [WIDTH-1:0] y_out;
    wire ready;
    
    reg signed [WIDTH-1:0] x [0:TIMESTEPS*LSTM_INPUTS-1];
    

    // INSTÂNCIA DA NETLIST

    lstm_network dut (
        .clk(clk),
        .reset(reset),
        .mode(mode),
        .clear(clear),
        .we(we),
        .addr(addr),
        .data_in(data_in),
        
        .\x[4][3] (x[4*4+3]), .\x[4][2] (x[4*4+2]), .\x[4][1] (x[4*4+1]), .\x[4][0] (x[4*4+0]),
        .\x[3][3] (x[3*4+3]), .\x[3][2] (x[3*4+2]), .\x[3][1] (x[3*4+1]), .\x[3][0] (x[3*4+0]),
        .\x[2][3] (x[2*4+3]), .\x[2][2] (x[2*4+2]), .\x[2][1] (x[2*4+1]), .\x[2][0] (x[2*4+0]),
        .\x[1][3] (x[1*4+3]), .\x[1][2] (x[1*4+2]), .\x[1][1] (x[1*4+1]), .\x[1][0] (x[1*4+0]),
        .\x[0][3] (x[0*4+3]), .\x[0][2] (x[0*4+2]), .\x[0][1] (x[0*4+1]), .\x[0][0] (x[0*4+0]),
        
        .y_out(y_out),
        .ready(ready)
    );
    
    reg [11:0] mem_addr [0:2047];
    reg signed [31:0] mem_data [0:2047];
    integer n_weights;
    
    reg [11:0] x_addr [0:TIMESTEPS*LSTM_INPUTS-1];
    reg signed [31:0] x_data [0:TIMESTEPS*LSTM_INPUTS-1];
    integer n_x_values;
    
    reg signed [31:0] python_result;
    integer ground_truth;
    
    integer m_total_samples    [0:NUM_FILES-1];
    integer m_python_errors    [0:NUM_FILES-1];
    integer m_verilog_errors   [0:NUM_FILES-1];
    real    m_python_accuracy  [0:NUM_FILES-1];
    real    m_verilog_accuracy [0:NUM_FILES-1];
    
    integer processed_files;
    real python_values [0:NUM_FILES-1];
    real verilog_values [0:NUM_FILES-1];
    
    integer results_file;
    
    integer i, t, f, file_idx, k;
    
    // ============================================================
    // FUNÇÕES
    // ============================================================

    function real q2real;
        input signed [31:0] q;
        begin
            q2real = q / 16777216.0;
        end
    endfunction
    
    function signed [31:0] real2q;
        input real r;
        begin
            real2q = $rtoi(r * 16777216.0);
        end
    endfunction
    
    function real calculate_r2;
        input integer n;
        real mean_actual, ss_tot, ss_res, diff_actual, diff_pred, sum;
        integer j;
        begin
            if (n < 2) begin
                calculate_r2 = 0.0;
            end
            else begin
                sum = 0.0;
                for (j = 0; j < n; j = j + 1)
                    sum = sum + python_values[j];
                mean_actual = sum / n;
                
                ss_tot = 0.0;
                ss_res = 0.0;
                for (j = 0; j < n; j = j + 1) begin
                    diff_actual = python_values[j] - mean_actual;
                    ss_tot = ss_tot + (diff_actual * diff_actual);
                    diff_pred = python_values[j] - verilog_values[j];
                    ss_res = ss_res + (diff_pred * diff_pred);
                end
                
                if (ss_tot != 0.0)
                    calculate_r2 = 1.0 - (ss_res / ss_tot);
                else
                    calculate_r2 = 0.0;
            end
        end
    endfunction
    
    // ============================================================
    // TASKS
    // ============================================================
    task clear_x;
        begin
            for (t = 0; t < TIMESTEPS; t = t + 1)
                for (f = 0; f < LSTM_INPUTS; f = f + 1)
                    x[t*LSTM_INPUTS + f] = 0;
        end
    endtask
    
    task load_data_from_file;
        input [1023:0] filename;
        output integer loaded_count;
        integer fd_x, fields, max_values;
        reg [1023:0] line;
        integer addr_val, data_val, python_val_int, gt_val;
        begin
            max_values = TIMESTEPS * LSTM_INPUTS;
            loaded_count = 0;
            n_x_values = 0;
            python_result = 0;
            ground_truth = 0;
            
            $display("");
            $display("LENDO dados de entrada de %0s", filename);
            
            fd_x = $fopen(filename, "r");
            if (fd_x == 0) begin
                $display("ERRO: Nao conseguiu abrir %0s!", filename);
                disable load_data_from_file;
            end
            
            while (!$feof(fd_x)) begin
                if ($fgets(line, fd_x) == 0) break;
                
                fields = $sscanf(line, "%h %h %h %d", addr_val, data_val, python_val_int, gt_val);
                if (fields == 4) begin
                    if (loaded_count == 0) begin
                        python_result = python_val_int;
                        ground_truth = gt_val;
                        $display("  Python result: 0x%08X (%f)", python_result, q2real(python_result));
                        $display("  Ground truth: %0d", ground_truth);
                    end
                    if (loaded_count < max_values) begin
                        x_addr[loaded_count] = addr_val[11:0];
                        x_data[loaded_count] = data_val;
                        loaded_count = loaded_count + 1;
                    end
                end
                else begin
                    fields = $sscanf(line, "%h %h", addr_val, data_val);
                    if (fields == 2) begin
                        if (loaded_count < max_values) begin
                            x_addr[loaded_count] = addr_val[11:0];
                            x_data[loaded_count] = data_val;
                            loaded_count = loaded_count + 1;
                        end
                    end
                end
            end
            
            $fclose(fd_x);
            n_x_values = loaded_count;
            $display("Li %0d valores de entrada (esperado: %0d)", n_x_values, max_values);
        end
    endtask
    
    task load_x_array;
        integer idx, timestep, feature;
        reg [11:0] addr_temp;
        begin
            for (idx = 0; idx < n_x_values; idx = idx + 1) begin
                addr_temp = x_addr[idx];
                timestep = addr_temp >> 2;
                feature = addr_temp & 3;
                if (timestep < TIMESTEPS && feature < LSTM_INPUTS)
                    x[timestep*LSTM_INPUTS + feature] = x_data[idx];
            end
        end
    endtask
    
    task reset_dut;
        begin
            reset = 1;
            clear = 1;
            mode = 0;
            we = 0;
            addr = 0;
            data_in = 0;
            #20;
            reset = 0;
            clear = 0;
            #20;
        end
    endtask
    
    task write_weights;
        begin
            $display("");
            $display("ESCREVENDO PESOS NO DUT");
            for (i = 0; i < n_weights; i = i + 1) begin
                @(posedge clk);
                we <= 1'b1;
                addr <= mem_addr[i];
                data_in <= mem_data[i];
            end
            @(posedge clk);
            we <= 1'b0;
            $display("Escritos %0d pesos", n_weights);
        end
    endtask
    
    task run_network;
        output signed [31:0] result;
        begin
            @(posedge clk);
            mode <= 1'b1;
            
            $display("EXECUTANDO LSTM NETWORK para arquivo %0d", file_idx);
            
            wait (ready == 1'b1);
            
            result = y_out;
            $display("  Resultado Verilog: %0d (Q8.24) = %.6f", result, q2real(result));
        end
    endtask
    
    task calculate_metrics;
        input real python_val, verilog_val;
        input integer gt;
        real python_class, verilog_class;
        integer python_correct, verilog_correct;
        begin
            python_class  = (python_val  >= THRESHOLD) ? 1.0 : 0.0;
            verilog_class = (verilog_val >= THRESHOLD) ? 1.0 : 0.0;
            
            python_correct  = (python_class  == gt) ? 1 : 0;
            verilog_correct = (verilog_class == gt) ? 1 : 0;
            
            m_total_samples[file_idx] = m_total_samples[file_idx] + 1;
            if (!python_correct)  m_python_errors[file_idx]  = m_python_errors[file_idx] + 1;
            if (!verilog_correct) m_verilog_errors[file_idx] = m_verilog_errors[file_idx] + 1;
            
            python_values[file_idx]  = python_val;
            verilog_values[file_idx] = verilog_val;
            
            if (results_file != 0)
                $fdisplay(results_file, "%f %f %0d", verilog_val, python_val, gt);
        end
    endtask
    
    task init_metrics;
        input integer idx;
        begin
            m_total_samples[idx]    = 0;
            m_python_errors[idx]    = 0;
            m_verilog_errors[idx]   = 0;
            m_python_accuracy[idx]  = 0.0;
            m_verilog_accuracy[idx] = 0.0;
        end
    endtask
    
    task generate_report;
        real total_python_errors, total_verilog_errors, total_samples;
        real r2_python_verilog;
        integer valid_samples, idx;
        begin
            valid_samples = 0;
            for (idx = 0; idx < processed_files; idx = idx + 1)
                if (m_total_samples[idx] > 0)
                    valid_samples = valid_samples + 1;
            
            r2_python_verilog = calculate_r2(valid_samples);
            
            $display("");
            $display("========================================");
            $display("RELATORIO FINAL");
            $display("========================================");
            
            total_python_errors  = 0.0;
            total_verilog_errors = 0.0;
            total_samples        = 0.0;
            
            for (idx = 0; idx < processed_files; idx = idx + 1) begin
                if (m_total_samples[idx] > 0) begin
                    m_python_accuracy[idx]  = 1.0 - (m_python_errors[idx]  * 1.0 / m_total_samples[idx]);
                    m_verilog_accuracy[idx] = 1.0 - (m_verilog_errors[idx] * 1.0 / m_total_samples[idx]);
                end
                
                $display("dados_%0d.mem     | Py: %6.2f%% | Ver: %6.2f%%",
                         idx,
                         m_python_accuracy[idx]  * 100.0,
                         m_verilog_accuracy[idx] * 100.0);
                
                total_python_errors  = total_python_errors  + m_python_errors[idx];
                total_verilog_errors = total_verilog_errors + m_verilog_errors[idx];
                total_samples        = total_samples        + m_total_samples[idx];
            end
            
            $display("------------------+-------------+-------------");
            $display("TOTAlS            | Py: %6.2f%% | Ver: %6.2f%%",
                     (total_samples - total_python_errors)  / total_samples * 100.0,
                     (total_samples - total_verilog_errors) / total_samples * 100.0);
            
            $display("");
            $display("  R2 (Verilog vs Python): %6.4f", r2_python_verilog);
            $display("");
            $display("  Total de amostras: %0d", total_samples);
            $display("  Erros Python:      %0d", total_python_errors);
            $display("  Erros Verilog:     %0d", total_verilog_errors);
            $display("========================================");
        end
    endtask
    
    // ============================================================
    // INITIAL
    // ============================================================
    initial begin
        integer fd_check;
        integer fd;
        reg [11:0] temp_addr;
        reg [31:0] temp_data;
        reg [1023:0] filename;
        integer loaded_count;
        reg signed [31:0] verilog_result;
        real python_val, verilog_val;
        
        $dumpfile("lstm_netlist_wave.vcd");
        $dumpvars(0, tb_lstm_network_netlist);
        
        results_file = $fopen("resultados simplest gate level p2.txt", "w");
        if (results_file == 0) begin
            $display("ERRO: Nao foi possivel criar resultados simplest gate level p2.txt");
            $finish;
        end
        $fdisplay(results_file, "# Verilog Python GroundTruth");
        
        // ---- CARREGA PESOS ----
        $display("");
        $display("LENDO weights.mem");
        
        n_weights = 0;
        fd = $fopen("weights.mem", "r");
        if (fd == 0) begin
            $display("ERRO: Nao conseguiu abrir weights.mem!");
            $finish;
        end
        
        while (!$feof(fd) && n_weights < 2048) begin
            if ($fscanf(fd, "%h %h", temp_addr, temp_data) == 2) begin
                mem_addr[n_weights] = temp_addr;
                mem_data[n_weights] = temp_data;
                n_weights = n_weights + 1;
            end
        end
        $fclose(fd);
        
        $display("Li %0d pesos do arquivo", n_weights);
        
        // ---- LOOP PRINCIPAL ----
        processed_files = 0;
        
        for (file_idx = 7781; file_idx < NUM_FILES; file_idx = file_idx + 1) begin
            $sformat(filename, "Small Global Processing/dados_%0d.mem", file_idx);
            
            fd_check = $fopen(filename, "r");
            if (fd_check == 0) begin
                $display("Arquivo %0s nao encontrado. Pulando...", filename);
                continue;
            end
            $fclose(fd_check);
            
            $display("");
            $display("PROCESSANDO ARQUIVO %0d: %0s", file_idx, filename);
            
            init_metrics(processed_files);
            clear_x;
            load_data_from_file(filename, loaded_count);
            
            if (loaded_count == 0) begin
                $display("ERRO: Nenhum dado carregado de %0s", filename);
                continue;
            end
            
            load_x_array;
            
            reset_dut;
            write_weights;      // <-- SEMPRE reescreve pesos
            
            run_network(verilog_result);
            
            python_val  = q2real(python_result);
            verilog_val = q2real(verilog_result);
            
            $display("");
            $display("  Python:  %.6f", python_val);
            $display("  Verilog: %.6f", verilog_val);
            $display("  GT:      %0d", ground_truth);
            
            calculate_metrics(python_val, verilog_val, ground_truth);
            
            processed_files = processed_files + 1;
        end
        
        $fclose(results_file);
        
        generate_report;
        
        $display("");
        $display("Simulacao completa!");
        $finish;
    end

endmodule
