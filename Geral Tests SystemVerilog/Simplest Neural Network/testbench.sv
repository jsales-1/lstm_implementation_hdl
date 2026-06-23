`timescale 1ns/1ps

module tb_lstm_network;

    parameter int WIDTH = 32;
    parameter int FRAC  = 16;
    
    parameter int LSTM_INPUTS  = 4;
    parameter int LSTM_HIDDEN  = 16;
    parameter int TIMESTEPS    = 20;
    parameter int RELU_INPUTS  = 16;
    parameter int RELU_NEURONS = 16;
    parameter int OUT_INPUTS   = 16;
    
    // Parâmetros para processamento múltiplo
    parameter int NUM_FILES = 40;          // Número de arquivos a processar (0 a NUM_FILES-1)
    parameter real THRESHOLD = 0.5;        // Limiar para classificação
    
    logic clk;
    logic reset;
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    logic mode;
    logic we;
    logic [20:0] addr;
    logic signed [WIDTH-1:0] data_in;
    logic signed [WIDTH-1:0] x [TIMESTEPS][LSTM_INPUTS];
    logic signed [WIDTH-1:0] y_out;
    logic ready;
    
    shortreal y_frac;
    
    lstm_network #(
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .LSTM_INPUTS(LSTM_INPUTS),
        .LSTM_HIDDEN(LSTM_HIDDEN),
        .TIMESTEPS(TIMESTEPS),
        .RELU_INPUTS(RELU_INPUTS),
        .RELU_NEURONS(RELU_NEURONS),
        .OUT_INPUTS(OUT_INPUTS)
    ) dut (
        .clk(clk),
        .reset(reset),
        .mode(mode),
        .we(we),
        .addr(addr),
        .data_in(data_in),
        .x(x),
        .y_out(y_out),
        .ready(ready)
    );
    
    // Arrays para pesos
    logic [20:0] mem_addr [0:2047];
    logic signed [31:0] mem_data [0:2047];
    int n_weights;
    
    // Arrays para dados de entrada - agora com tamanho suficiente para todos os timesteps
    logic [20:0] x_addr [0:TIMESTEPS*LSTM_INPUTS-1];
    logic signed [31:0] x_data [0:TIMESTEPS*LSTM_INPUTS-1];
    int n_x_values;
    
    // Arrays para resultados do Python (lidos do arquivo)
    logic signed [31:0] python_result;
    int ground_truth;
    
    // Estruturas para estatísticas
    typedef struct {
        int total_samples;
        int python_errors;
        int verilog_errors;
        int both_correct;
        int both_wrong;
        int python_wrong_verilog_correct;
        int python_correct_verilog_wrong;
        real python_accuracy;
        real verilog_accuracy;
        string filename;
    } metrics_t;
    
    metrics_t file_metrics [0:NUM_FILES-1];
    int processed_files;
    
    // Arquivo de resultados (apenas este, sem internal_outputs)
    int results_file;
    
    integer i, t, f, file_idx;
    
    function real q2real;
        input signed [31:0] q;
        q2real = q / 65536.0;
    endfunction
    
    function signed [31:0] real2q;
        input real r;
        real2q = $rtoi(r * 65536.0);
    endfunction
    
    // Função para limpar a matriz x
    task clear_x();
        for (t = 0; t < TIMESTEPS; t++) begin
            for (f = 0; f < LSTM_INPUTS; f++) begin
                x[t][f] = 0;
            end
        end
    endtask
    
    // ============================================================
    // LEITURA ROBUSTA LINHA A LINHA (agora com capacidade total)
    // ============================================================
    task load_data_from_file(input string filename, output int loaded_count);
        int fd_x;
        string line;
        int addr_val, data_val, python_val_int, gt_val;
        int fields;
        int max_values = TIMESTEPS * LSTM_INPUTS;   // 80
        
        loaded_count = 0;
        n_x_values = 0;
        python_result = 0;
        ground_truth = 0;
        
        $display("");
        $display("LENDO dados de entrada de %s", filename);
        
        fd_x = $fopen(filename, "r");
        if (fd_x == 0) begin
            $display("ERRO: Não conseguiu abrir %s!", filename);
            return;
        end
        
        // Lê linha por linha
        while (!$feof(fd_x)) begin
            if ($fgets(line, fd_x) == 0) break;
            line = line.substr(0, line.len()-1); // remove newline
            
            // Tenta parsear com 4 campos (primeira linha especial)
            fields = $sscanf(line, "%h %h %h %d", addr_val, data_val, python_val_int, gt_val);
            if (fields == 4) begin
                if (loaded_count == 0) begin
                    python_result = python_val_int;
                    ground_truth = gt_val;
                    $display("  Python result: 0x%08X (%f)", python_result, q2real(python_result));
                    $display("  Ground truth: %d", ground_truth);
                end
                // Armazena o dado de entrada (segundo campo)
                if (loaded_count < max_values) begin
                    x_addr[loaded_count] = addr_val;
                    x_data[loaded_count] = data_val;
                    loaded_count++;
                end
            end
            else begin
                // Tenta parsear com 2 campos (linhas normais)
                fields = $sscanf(line, "%h %h", addr_val, data_val);
                if (fields == 2) begin
                    if (loaded_count < max_values) begin
                        x_addr[loaded_count] = addr_val;
                        x_data[loaded_count] = data_val;
                        loaded_count++;
                    end
                end
                // senão, linha vazia ou formato inesperado -> ignora
            end
        end
        
        $fclose(fd_x);
        n_x_values = loaded_count;
        $display("Li %0d valores de entrada (esperado: %0d)", n_x_values, max_values);
        if (n_x_values != max_values) begin
            $display("ATENÇÃO: Número de valores lido difere do esperado!");
        end
    endtask
    
    // Função para carregar dados no array x
    task load_x_array();
        for (int idx = 0; idx < n_x_values; idx++) begin
            logic [20:0] addr_temp;
            int timestep, feature;
            
            addr_temp = x_addr[idx];
            timestep = addr_temp >> 2;   // endereço / 4
            feature = addr_temp & 3;     // resto da divisão por 4
            
            if (timestep < TIMESTEPS && feature < LSTM_INPUTS) begin
                x[timestep][feature] = x_data[idx];
            end
        end
    endtask
    
    // Função para resetar o DUT
    task reset_dut();
        reset = 1;
        mode = 0;
        we = 0;
        addr = 0;
        data_in = 0;
        #20;
        reset = 0;
        #20;
    endtask
    
    // Função para escrever pesos no DUT
    task write_weights();
        $display("");
        $display("ESCREVENDO PESOS NO DUT");
        
        for (i = 0; i < n_weights; i++) begin
            @(posedge clk);
            we <= 1'b1;
            addr <= mem_addr[i];
            data_in <= mem_data[i];
        end
        
        @(posedge clk);
        we <= 1'b0;
        $display("Escritos %0d pesos", n_weights);
    endtask
    
    // Função para executar a rede e retornar resultado
    task run_network(output logic signed [31:0] result);
        @(posedge clk);
        mode <= 1'b1;
        
        $display("EXECUTANDO LSTM NETWORK para arquivo %0d", file_idx);
        
        wait (ready == 1'b1);
        
        result = y_out;
        y_frac = q2real(y_out);
        
        $display("  Resultado Verilog: %0d (Q16.16) = %.6f (real)", y_out, y_frac);
    endtask
    
    // Função para calcular métricas
    task calculate_metrics(
        input string filename,
        input real python_val,
        input real verilog_val,
        input int ground_truth,
        input int sample_idx
    );
        real python_class, verilog_class;
        int python_correct, verilog_correct;
        
        python_class = (python_val >= THRESHOLD) ? 1.0 : 0.0;
        verilog_class = (verilog_val >= THRESHOLD) ? 1.0 : 0.0;
        
        python_correct = (python_class == ground_truth) ? 1 : 0;
        verilog_correct = (verilog_class == ground_truth) ? 1 : 0;
        
        // Atualiza métricas
        file_metrics[file_idx].total_samples++;
        
        if (!python_correct) file_metrics[file_idx].python_errors++;
        if (!verilog_correct) file_metrics[file_idx].verilog_errors++;
        
        if (python_correct && verilog_correct) begin
            file_metrics[file_idx].both_correct++;
        end else if (!python_correct && !verilog_correct) begin
            file_metrics[file_idx].both_wrong++;
        end else if (!python_correct && verilog_correct) begin
            file_metrics[file_idx].python_wrong_verilog_correct++;
        end else begin
            file_metrics[file_idx].python_correct_verilog_wrong++;
        end
        
        // Escreve no arquivo de resultados
        if (results_file != 0) begin
            $fdisplay(results_file, "%f %f %d", verilog_val, python_val, ground_truth);
        end
    endtask
    
    // Função para inicializar métricas
    task init_metrics(input int idx, input string fname);
        file_metrics[idx].filename = fname;
        file_metrics[idx].total_samples = 0;
        file_metrics[idx].python_errors = 0;
        file_metrics[idx].verilog_errors = 0;
        file_metrics[idx].both_correct = 0;
        file_metrics[idx].both_wrong = 0;
        file_metrics[idx].python_wrong_verilog_correct = 0;
        file_metrics[idx].python_correct_verilog_wrong = 0;
        file_metrics[idx].python_accuracy = 0.0;
        file_metrics[idx].verilog_accuracy = 0.0;
    endtask
    
    // Função para gerar relatório final
    task generate_report();
        real total_python_errors, total_verilog_errors;
        real total_samples;
        
        $display("");
        $display("========================================");
        $display("RELATÓRIO FINAL DE VALIDAÇÃO");
        $display("========================================");
        $display("");
        
        $display("Arquivo           | Acert. Python | Acert. Verilog | Ambos Corretos | Ambos Errados | Py Err/Ver Corr | Py Corr/Ver Err");
        $display("------------------+---------------+---------------+----------------+---------------+-----------------+----------------");
        
        total_python_errors = 0;
        total_verilog_errors = 0;
        total_samples = 0;
        
        for (int idx = 0; idx < processed_files; idx++) begin
            if (file_metrics[idx].total_samples > 0) begin
                file_metrics[idx].python_accuracy = 1.0 - (file_metrics[idx].python_errors / file_metrics[idx].total_samples);
                file_metrics[idx].verilog_accuracy = 1.0 - (file_metrics[idx].verilog_errors / file_metrics[idx].total_samples);
            end
            
            $display("%-16s | %13.2f%% | %13.2f%% | %13d | %13d | %15d | %15d",
                     file_metrics[idx].filename,
                     file_metrics[idx].python_accuracy * 100.0,
                     file_metrics[idx].verilog_accuracy * 100.0,
                     file_metrics[idx].both_correct,
                     file_metrics[idx].both_wrong,
                     file_metrics[idx].python_wrong_verilog_correct,
                     file_metrics[idx].python_correct_verilog_wrong);
            
            total_python_errors += file_metrics[idx].python_errors;
            total_verilog_errors += file_metrics[idx].verilog_errors;
            total_samples += file_metrics[idx].total_samples;
        end
        
        $display("------------------+---------------+---------------+----------------+---------------+-----------------+----------------");
        $display("TOTAIS            | %13.2f%% | %13.2f%% | %13d | %13d | %15d | %15d",
                  (total_samples - total_python_errors) / total_samples * 100.0,
                  (total_samples - total_verilog_errors) / total_samples * 100.0,
                  0, 0, 0, 0);
        
        $display("");
        $display("RESUMO:");
        $display("  Total de amostras: %0d", total_samples);
        $display("  Erros Python: %0d (%.2f%%)", total_python_errors, total_python_errors/total_samples*100.0);
        $display("  Erros Verilog: %0d (%.2f%%)", total_verilog_errors, total_verilog_errors/total_samples*100.0);
        $display("");
        $display("Arquivo de resultados detalhados: resultados.txt");
        $display("========================================");
    endtask
    
    initial begin
        int fd_check;
        
        $dumpfile("lstm_wave.vcd");
        $dumpvars(0, tb_lstm_network);
        
        // Abre arquivo de resultados (apenas este)
        results_file = $fopen("resultados.txt", "w");
        if (results_file == 0) begin
            $display("ERRO: Não foi possível criar resultados.txt");
            $finish;
        end
        $fdisplay(results_file, "# Verilog Python GroundTruth");
        
        // =============================================
        // CARREGA PESOS UMA ÚNICA VEZ
        // =============================================
        $display("");
        $display("LENDO weights.mem");
        
        n_weights = 0;
        begin
            int fd;
            logic [20:0] temp_addr;
            logic [31:0] temp_data;
            
            fd = $fopen("weights.mem", "r");
            if (fd == 0) begin
                $display("ERRO: Não conseguiu abrir weights.mem!");
                $finish;
            end
            
            while (!$feof(fd) && n_weights < 2048) begin
                if ($fscanf(fd, "%h %h", temp_addr, temp_data) == 2) begin
                    mem_addr[n_weights] = temp_addr;
                    mem_data[n_weights] = temp_data;
                    n_weights++;
                end
            end
            $fclose(fd);
        end
        
        $display("Li %0d pesos do arquivo", n_weights);
        
        // =============================================
        // LOOP PRINCIPAL: PROCESSAR MÚLTIPLOS ARQUIVOS
        // =============================================
        processed_files = 0;
        
        for (file_idx = 0; file_idx < NUM_FILES; file_idx++) begin
            string filename;
            int loaded_count;
            logic signed [31:0] verilog_result;
            real python_val, verilog_val;
            
            $sformat(filename, "dados_%0d.mem", file_idx);
            
            fd_check = $fopen(filename, "r");
            if (fd_check == 0) begin
                $display("Arquivo %s não encontrado. Pulando...", filename);
                continue;
            end
            $fclose(fd_check);
            
            $display("");
            $display("========================================");
            $display("PROCESSANDO ARQUIVO %0d: %s", file_idx, filename);
            $display("========================================");
            
            init_metrics(processed_files, filename);
            
            clear_x();
            
            load_data_from_file(filename, loaded_count);
            if (loaded_count == 0) begin
                $display("ERRO: Nenhum dado carregado de %s", filename);
                continue;
            end
            
            load_x_array();
            
            /*$display("");
            $display("Valores de entrada carregados (primeiros 5 timesteps):");
            for (t = 0; t < (TIMESTEPS > 5 ? 5 : TIMESTEPS); t++) begin
                $display("  x[%0d]: [%0d, %0d, %0d, %0d]", 
                         t, x[t][0], x[t][1], x[t][2], x[t][3]);
            end
            if (TIMESTEPS > 5) $display("  ... (mais timesteps)");
            */
            reset_dut();
            write_weights();
            run_network(verilog_result);
            
            python_val = q2real(python_result);
            verilog_val = q2real(verilog_result);
            
            $display("");
            $display("  Python result: %.6f", python_val);
            $display("  Verilog result: %.6f", verilog_val);
            $display("  Ground truth: %d", ground_truth);
            
            calculate_metrics(filename, python_val, verilog_val, ground_truth, 0);
            
            processed_files++;
        end
        
        $fclose(results_file);
        
        generate_report();
        
        $display("");
        $display("Simulação completa!");
        $finish;
    end
    
endmodule