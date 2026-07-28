
`timescale 1ns/1ps

module tb_lstm_network;

    parameter int WIDTH = 32;
    parameter int FRAC  = 24;
    
    parameter int LSTM_INPUTS  = 4;
    parameter int LSTM_HIDDEN  = 16;
    parameter int TIMESTEPS    = 120;
    parameter int RELU_INPUTS  = 16;
    parameter int RELU_NEURONS = 32;
    parameter int OUT_INPUTS   = 32;
    
    // Parâmetros para processamento múltiplo
    parameter int NUM_FILES = 5;
    parameter real THRESHOLD = 0.5;
    
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
    
    // Arquivo para salvar saída da LSTM
    int lstm_results_file;
    
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
    
    // Arrays para dados de entrada
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
    
    // Arrays para cálculo do R²
    real python_values [0:NUM_FILES-1];
    real verilog_values [0:NUM_FILES-1];
    
    // Arquivo de resultados
    int results_file;
    
    integer i, t, f, file_idx;
    
    function real q2real;
        input signed [31:0] q;
        q2real = q / 16777216.0;
    endfunction
    
    function signed [31:0] real2q;
        input real r;
        real2q = $rtoi(r * 16777216.0);
    endfunction
    
    // ============================================================
    // FUNÇÃO PARA SALVAR SAÍDA DA LSTM EM ARQUIVO
    // ============================================================
    task save_lstm_outputs();
        real lstm_val;
        
        lstm_results_file = $fopen("resultados_lstm.txt", "a");
        if (lstm_results_file == 0) begin
            $display("ERRO: Não foi possível abrir resultados_lstm.txt");
            return;
        end
        
        if (file_idx == 0) begin
            $fdisplay(lstm_results_file, "# Saída da LSTM - Valores em float32 (Q8.24)");
            $fdisplay(lstm_results_file, "# Formato: [Arquivo] [Neuron] [Value]");
            $fdisplay(lstm_results_file, "#----------------------------------------");
        end
        
        for (int idx = 0; idx < LSTM_HIDDEN; idx++) begin
            lstm_val = q2real(dut.lstm_h_out[TIMESTEPS-1][idx]);
            $fdisplay(lstm_results_file, "%0d %d %f", file_idx, idx, lstm_val);
        end
        
        $fclose(lstm_results_file);
    endtask
    
    // Função para limpar a matriz x
    task clear_x();
        for (t = 0; t < TIMESTEPS; t++) begin
            for (f = 0; f < LSTM_INPUTS; f++) begin
                x[t][f] = 0;
            end
        end
    endtask
    
    // ============================================================
    // LEITURA ROBUSTA LINHA A LINHA
    // ============================================================
    task load_data_from_file(input string filename, output int loaded_count);
        int fd_x;
        string line;
        int addr_val, data_val, python_val_int, gt_val;
        int fields;
        int max_values;
        
        max_values = TIMESTEPS * LSTM_INPUTS;
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
        
        while (!$feof(fd_x)) begin
            if ($fgets(line, fd_x) == 0) break;
            line = line.substr(0, line.len()-1);
            
            fields = $sscanf(line, "%h %h %h %d", addr_val, data_val, python_val_int, gt_val);
            if (fields == 4) begin
                if (loaded_count == 0) begin
                    python_result = python_val_int;
                    ground_truth = gt_val;
                    $display("  Python result: 0x%08X (%f)", python_result, q2real(python_result));
                    $display("  Ground truth: %d", ground_truth);
                end
                if (loaded_count < max_values) begin
                    x_addr[loaded_count] = addr_val;
                    x_data[loaded_count] = data_val;
                    loaded_count++;
                end
            end
            else begin
                fields = $sscanf(line, "%h %h", addr_val, data_val);
                if (fields == 2) begin
                    if (loaded_count < max_values) begin
                        x_addr[loaded_count] = addr_val;
                        x_data[loaded_count] = data_val;
                        loaded_count++;
                    end
                end
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
            timestep = addr_temp >> 2;
            feature = addr_temp & 3;
            
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
        
        $display("  Resultado Verilog: %0d (Q8.24) = %.6f (real)", y_out, y_frac);
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
        
        python_values[file_idx] = python_val;
        verilog_values[file_idx] = verilog_val;
        
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
    
    // ============================================================
    // FUNÇÃO PARA CALCULAR R²
    // ============================================================
    function automatic real calculate_r2(
        ref real actual[0:NUM_FILES-1],
        ref real predicted[0:NUM_FILES-1],
        int n
    );
        real mean_actual;
        real ss_tot;
        real ss_res;
        real diff_actual;
        real diff_pred;
        int j;
        
        mean_actual = 0.0;
        ss_tot = 0.0;
        ss_res = 0.0;
        calculate_r2 = 0.0;
        
        if (n < 2) begin
            calculate_r2 = 0.0;
            return calculate_r2;
        end
        
        for (j = 0; j < n; j++) begin
            mean_actual = mean_actual + actual[j];
        end
        mean_actual = mean_actual / n;
        
        for (j = 0; j < n; j++) begin
            diff_actual = actual[j] - mean_actual;
            ss_tot = ss_tot + (diff_actual * diff_actual);
            
            diff_pred = actual[j] - predicted[j];
            ss_res = ss_res + (diff_pred * diff_pred);
        end
        
        if (ss_tot != 0.0) begin
            calculate_r2 = 1.0 - (ss_res / ss_tot);
        end else begin
            calculate_r2 = 0.0;
        end
        
        return calculate_r2;
    endfunction
    
    // Função para gerar relatório final
    task generate_report();
        real total_python_errors, total_verilog_errors;
        real total_samples;
        real r2_python_verilog;
        
        real python_vals [0:NUM_FILES-1];
        real verilog_vals [0:NUM_FILES-1];
        int valid_samples;
        int idx;
        
        for (idx = 0; idx < NUM_FILES; idx++) begin
            python_vals[idx] = 0.0;
            verilog_vals[idx] = 0.0;
        end
        
        valid_samples = 0;
        for (idx = 0; idx < processed_files; idx++) begin
            if (file_metrics[idx].total_samples > 0) begin
                python_vals[valid_samples] = python_values[idx];
                verilog_vals[valid_samples] = verilog_values[idx];
                valid_samples++;
            end
        end
        
        r2_python_verilog = calculate_r2(python_vals, verilog_vals, valid_samples);
        
        $display("");
       
        $display("RELATÓRIO FINAL DE VALIDAÇÃO");
       
        $display("");
        
        $display("Arquivo           | Acert. Python | Acert. Verilog | Ambos Corretos | Ambos Errados | Py Err/Ver Corr | Py Corr/Ver Err");
        $display("------------------+---------------+---------------+----------------+---------------+-----------------+----------------");
        
        total_python_errors = 0.0;
        total_verilog_errors = 0.0;
        total_samples = 0.0;
        
        for (idx = 0; idx < processed_files; idx++) begin
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
            
            total_python_errors = total_python_errors + file_metrics[idx].python_errors;
            total_verilog_errors = total_verilog_errors + file_metrics[idx].verilog_errors;
            total_samples = total_samples + file_metrics[idx].total_samples;
        end
        
        $display("------------------+---------------+---------------+----------------+---------------+-----------------+----------------");
        $display("TOTAIS            | %13.2f%% | %13.2f%% | %13d | %13d | %15d | %15d",
                  (total_samples - total_python_errors) / total_samples * 100.0,
                  (total_samples - total_verilog_errors) / total_samples * 100.0,
                  0, 0, 0, 0);
        
        $display("");
       
        $display("MÉTRICA DE REGRESSÃO (Verilog vs Python)");
       
        $display("");
        $display("  R² (Coeficiente de Determinação): %6.4f", r2_python_verilog);
        $display("");
        $display("  Interpretação do R²:");
        if (r2_python_verilog >= 0.9) begin
            $display("    ✅ Excelente correlação (R² ≥ 0.9)");
        end else if (r2_python_verilog >= 0.7) begin
            $display("    ✅ Boa correlação (R² ≥ 0.7)");
        end else if (r2_python_verilog >= 0.5) begin
            $display("    ⚠️ Correlação moderada (R² ≥ 0.5)");
        end else begin
            $display("    ❌ Baixa correlação (R² < 0.5) - Verificar implementação");
        end
        $display("");
        
       
        $display("RESUMO DE CLASSIFICAÇÃO");
       
        $display("");
        $display("  Total de amostras: %0d", total_samples);
        $display("  Erros Python:      %0d (%.2f%%)", total_python_errors, total_python_errors/total_samples*100.0);
        $display("  Erros Verilog:     %0d (%.2f%%)", total_verilog_errors, total_verilog_errors/total_samples*100.0);
        $display("  Acurácia Verilog:  %.2f%%", (total_samples - total_verilog_errors) / total_samples * 100.0);
        $display("");
        $display("Arquivo de resultados detalhados: resultados.txt");
        $display("Arquivo com saída LSTM: resultados_lstm.txt");
       
    endtask
    
    initial begin
        int fd_check;
        
        $dumpfile("lstm_wave.vcd");
        $dumpvars(0, tb_lstm_network);
        
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
           
            $display("PROCESSANDO ARQUIVO %0d: %s", file_idx, filename);

            
            init_metrics(processed_files, filename);
            
            clear_x();
            
            load_data_from_file(filename, loaded_count);
            if (loaded_count == 0) begin
                $display("ERRO: Nenhum dado carregado de %s", filename);
                continue;
            end
            
            load_x_array();
            
            reset_dut();
            if(file_idx == 0) begin
                write_weights();
            end
            run_network(verilog_result);
            
            python_val = q2real(python_result);
            verilog_val = q2real(verilog_result);
            
            $display("");
            $display("  Python result: %.6f", python_val);
            $display("  Verilog result: %.6f", verilog_val);
            $display("  Ground truth: %d", ground_truth);
            
            // ============================================================
            // SALVAR SAÍDA DA LSTM EM ARQUIVO (todos os arquivos)
            // ============================================================
            save_lstm_outputs();
            
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