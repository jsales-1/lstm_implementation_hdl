`timescale 1ns/1ps

module tb_lstm_network;

    parameter int WIDTH = 32;
    parameter int FRAC  = 16;
    
    parameter int LSTM_INPUTS  = 4;
    parameter int LSTM_HIDDEN  = 16;
    parameter int TIMESTEPS    = 120;
    parameter int RELU_INPUTS  = 16;
    parameter int RELU_NEURONS = 32;
    parameter int OUT_INPUTS   = 32;
    
    logic clk;
    logic reset;
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    logic mode;  // 0 = WRITE | 1 = RUN
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
    
    logic [20:0] mem_addr [0:2047];
    logic signed [31:0] mem_data [0:2047];
    int n_weights;
    
    integer i, t, f;
    
    function real q2real;
        input signed [31:0] q;
        q2real = q / 65536.0;
    endfunction
    
    function signed [31:0] real2q;
        input real r;
        real2q = $rtoi(r * 65536.0);
    endfunction
    
    initial begin
        $dumpfile("lstm_wave.vcd");
        $dumpvars(0, tb_lstm_network);
        
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
        
        for (i = 0; i < 5 && i < n_weights; i++) begin
            $display("  [%0d] addr=0x%05h data=%0d (%.6f real)", 
                     i, mem_addr[i], mem_data[i], q2real(mem_data[i]));
        end
        
        // INIT
        reset   = 1;
        mode    = 0;
        we      = 0;
        addr    = 0;
        data_in = 0;
        
        for (t = 0; t < TIMESTEPS; t++) begin
            for (f = 0; f < LSTM_INPUTS; f++) begin
                x[t][f] = 0;
            end
        end
        
        #20;
        reset = 0;
        #20;
        
        $display("");
        $display("ESCREVENDO PESOS NO DUT");
        
        for (i = 0; i < n_weights; i++) begin
            @(posedge clk);
            we      <= 1'b1;
            addr    <= mem_addr[i];
            data_in <= mem_data[i];
        end
        
        @(posedge clk);
        we <= 1'b0;
        
        $display("Escritos %0d pesos", n_weights);
      
 x[0][0] = -32'sd516;   x[0][1] = -32'sd146;   x[0][2] =  32'sd58;   x[0][3] = -32'sd102;
x[1][0] = -32'sd516;   x[1][1] = -32'sd146;   x[1][2] =  32'sd58;   x[1][3] = -32'sd102;
x[2][0] = -32'sd516;   x[2][1] = -32'sd146;   x[2][2] =  32'sd58;   x[2][3] = -32'sd102;
x[3][0] = -32'sd516;   x[3][1] = -32'sd146;   x[3][2] =  32'sd58;   x[3][3] = -32'sd102;
x[4][0] = -32'sd516;   x[4][1] = -32'sd146;   x[4][2] =  32'sd58;   x[4][3] = -32'sd102;
x[5][0] = -32'sd516;   x[5][1] = -32'sd146;   x[5][2] =  32'sd58;   x[5][3] = -32'sd102;
x[6][0] = -32'sd516;   x[6][1] = -32'sd146;   x[6][2] =  32'sd58;   x[6][3] = -32'sd102;
x[7][0] = -32'sd516;   x[7][1] = -32'sd146;   x[7][2] =  32'sd58;   x[7][3] = -32'sd102;
x[8][0] = -32'sd516;   x[8][1] = -32'sd146;   x[8][2] =  32'sd58;   x[8][3] = -32'sd102;
x[9][0] = -32'sd516;   x[9][1] = -32'sd146;   x[9][2] =  32'sd58;   x[9][3] = -32'sd102;
x[10][0] = -32'sd516;   x[10][1] = -32'sd146;   x[10][2] =  32'sd58;   x[10][3] = -32'sd102;
x[11][0] =  32'sd782;   x[11][1] =  32'sd2524;   x[11][2] =  32'sd2244;   x[11][3] = -32'sd3044;
x[12][0] = -32'sd196;   x[12][1] =  32'sd3262;   x[12][2] = -32'sd1335;   x[12][3] =  32'sd540;
x[13][0] =  32'sd1569;   x[13][1] = -32'sd284;   x[13][2] =  32'sd881;   x[13][3] = -32'sd1146;
x[14][0] = -32'sd4605;   x[14][1] =  32'sd3147;   x[14][2] =  32'sd288;   x[14][3] = -32'sd5871;
x[15][0] =  32'sd977;   x[15][1] = -32'sd3901;   x[15][2] = -32'sd1364;   x[15][3] =  32'sd1893;
x[16][0] = -32'sd632;   x[16][1] = -32'sd1019;   x[16][2] = -32'sd1829;   x[16][3] =  32'sd1068;
x[17][0] =  32'sd160;   x[17][1] = -32'sd1696;   x[17][2] = -32'sd939;   x[17][3] = -32'sd1085;
x[18][0] = -32'sd810;   x[18][1] =  32'sd3;   x[18][2] =  32'sd890;   x[18][3] = -32'sd911;
x[19][0] = -32'sd4470;   x[19][1] =  32'sd4398;   x[19][2] =  32'sd2868;   x[19][3] = -32'sd166;
x[20][0] =  32'sd521;   x[20][1] =  32'sd406;   x[20][2] =  32'sd207;   x[20][3] = -32'sd945;
x[21][0] =  32'sd611;   x[21][1] = -32'sd21;   x[21][2] =  32'sd1245;   x[21][3] =  32'sd2780;
x[22][0] =  32'sd1322;   x[22][1] = -32'sd360;   x[22][2] = -32'sd518;   x[22][3] =  32'sd227;
x[23][0] =  32'sd512;   x[23][1] =  32'sd2074;   x[23][2] = -32'sd3629;   x[23][3] = -32'sd429;
x[24][0] =  32'sd4504;   x[24][1] = -32'sd542;   x[24][2] =  32'sd2292;   x[24][3] =  32'sd327;
x[25][0] =  32'sd500;   x[25][1] = -32'sd617;   x[25][2] =  32'sd4064;   x[25][3] = -32'sd2201;
x[26][0] =  32'sd3459;   x[26][1] = -32'sd1861;   x[26][2] = -32'sd254;   x[26][3] = -32'sd2802;
x[27][0] = -32'sd3544;   x[27][1] = -32'sd369;   x[27][2] =  32'sd2337;   x[27][3] = -32'sd3944;
x[28][0] = -32'sd7098;   x[28][1] =  32'sd8534;   x[28][2] =  32'sd3029;   x[28][3] = -32'sd8428;
x[29][0] = -32'sd1170;   x[29][1] = -32'sd1269;   x[29][2] = -32'sd2614;   x[29][3] =  32'sd383;
x[30][0] = -32'sd3080;   x[30][1] =  32'sd2280;   x[30][2] =  32'sd2180;   x[30][3] = -32'sd5470;
x[31][0] =  32'sd500;   x[31][1] = -32'sd617;   x[31][2] =  32'sd4064;   x[31][3] = -32'sd2201;
x[32][0] = -32'sd6087;   x[32][1] =  32'sd3563;   x[32][2] =  32'sd5204;   x[32][3] =  32'sd193;
x[33][0] = -32'sd978;   x[33][1] = -32'sd291;   x[33][2] =  32'sd554;   x[33][3] = -32'sd1164;
x[34][0] = -32'sd2359;   x[34][1] =  32'sd1635;   x[34][2] = -32'sd1946;   x[34][3] =  32'sd723;
x[35][0] =  32'sd5832;   x[35][1] = -32'sd3475;   x[35][2] = -32'sd4616;   x[35][3] =  32'sd1535;
x[36][0] = -32'sd3479;   x[36][1] =  32'sd1993;   x[36][2] =  32'sd4010;   x[36][3] = -32'sd435;
x[37][0] =  32'sd481;   x[37][1] =  32'sd2981;   x[37][2] =  32'sd275;   x[37][3] = -32'sd821;
x[38][0] = -32'sd727;   x[38][1] =  32'sd798;   x[38][2] =  32'sd2043;   x[38][3] =  32'sd524;
x[39][0] = -32'sd662;   x[39][1] = -32'sd2276;   x[39][2] = -32'sd1975;   x[39][3] =  32'sd222;
x[40][0] =  32'sd6033;   x[40][1] =  32'sd1205;   x[40][2] = -32'sd1566;   x[40][3] =  32'sd5214;
x[41][0] = -32'sd303;   x[41][1] = -32'sd268;   x[41][2] =  32'sd12;   x[41][3] =  32'sd46;
x[42][0] =  32'sd4716;   x[42][1] = -32'sd2058;   x[42][2] = -32'sd1887;   x[42][3] =  32'sd4866;
x[43][0] =  32'sd1662;   x[43][1] = -32'sd3429;   x[43][2] = -32'sd1893;   x[43][3] =  32'sd3247;
x[44][0] = -32'sd1410;   x[44][1] =  32'sd817;   x[44][2] =  32'sd925;   x[44][3] = -32'sd2592;
x[45][0] = -32'sd4470;   x[45][1] =  32'sd4398;   x[45][2] =  32'sd2868;   x[45][3] = -32'sd166;
x[46][0] =  32'sd1132;   x[46][1] =  32'sd1517;   x[46][2] = -32'sd885;   x[46][3] = -32'sd139;
x[47][0] = -32'sd595;   x[47][1] =  32'sd2775;   x[47][2] =  32'sd1821;   x[47][3] =  32'sd854;
x[48][0] = -32'sd132;   x[48][1] = -32'sd291;   x[48][2] = -32'sd772;   x[48][3] =  32'sd346;
x[49][0] =  32'sd1296;   x[49][1] = -32'sd2099;   x[49][2] = -32'sd639;   x[49][3] =  32'sd2295;
x[50][0] =  32'sd3815;   x[50][1] = -32'sd4080;   x[50][2] = -32'sd3350;   x[50][3] =  32'sd4303;
x[51][0] =  32'sd366;   x[51][1] = -32'sd918;   x[51][2] = -32'sd427;   x[51][3] =  32'sd1355;
x[52][0] =  32'sd1792;   x[52][1] =  32'sd1211;   x[52][2] =  32'sd892;   x[52][3] =  32'sd2302;
x[53][0] = -32'sd810;   x[53][1] =  32'sd3;   x[53][2] =  32'sd890;   x[53][3] = -32'sd911;
x[54][0] = -32'sd1333;   x[54][1] =  32'sd1046;   x[54][2] = -32'sd4149;   x[54][3] =  32'sd1306;
x[55][0] =  32'sd85;   x[55][1] = -32'sd2204;   x[55][2] =  32'sd243;   x[55][3] =  32'sd1086;
x[56][0] = -32'sd2559;   x[56][1] =  32'sd2281;   x[56][2] =  32'sd5109;   x[56][3] = -32'sd1395;
x[57][0] = -32'sd2296;   x[57][1] =  32'sd3900;   x[57][2] = -32'sd265;   x[57][3] =  32'sd216;
x[58][0] = -32'sd810;   x[58][1] =  32'sd3;   x[58][2] =  32'sd890;   x[58][3] = -32'sd911;
x[59][0] = -32'sd2346;   x[59][1] =  32'sd4275;   x[59][2] =  32'sd2724;   x[59][3] = -32'sd3332;
x[60][0] = -32'sd17;   x[60][1] = -32'sd1804;   x[60][2] = -32'sd1553;   x[60][3] = -32'sd306;
x[61][0] = -32'sd913;   x[61][1] =  32'sd1783;   x[61][2] =  32'sd1064;   x[61][3] = -32'sd317;
x[62][0] =  32'sd4567;   x[62][1] = -32'sd5955;   x[62][2] = -32'sd4386;   x[62][3] =  32'sd1997;
x[63][0] = -32'sd3264;   x[63][1] =  32'sd3800;   x[63][2] = -32'sd666;   x[63][3] =  32'sd1875;
x[64][0] =  32'sd933;   x[64][1] = -32'sd1018;   x[64][2] = -32'sd1417;   x[64][3] =  32'sd1864;
x[65][0] = -32'sd810;   x[65][1] =  32'sd3;   x[65][2] =  32'sd890;   x[65][3] = -32'sd911;
x[66][0] = -32'sd1487;   x[66][1] = -32'sd3929;   x[66][2] = -32'sd2731;   x[66][3] =  32'sd3838;
x[67][0] = -32'sd7288;   x[67][1] =  32'sd7055;   x[67][2] =  32'sd7632;   x[67][3] = -32'sd5912;
x[68][0] =  32'sd881;   x[68][1] =  32'sd540;   x[68][2] =  32'sd8;   x[68][3] =  32'sd3052;
x[69][0] =  32'sd869;   x[69][1] = -32'sd1998;   x[69][2] = -32'sd2166;   x[69][3] = -32'sd2951;
x[70][0] =  32'sd2271;   x[70][1] = -32'sd540;   x[70][2] =  32'sd950;   x[70][3] = -32'sd644;
x[71][0] =  32'sd1443;   x[71][1] = -32'sd2237;   x[71][2] = -32'sd1308;   x[71][3] =  32'sd4788;
x[72][0] =  32'sd3733;   x[72][1] = -32'sd4761;   x[72][2] = -32'sd1937;   x[72][3] =  32'sd4719;
x[73][0] = -32'sd810;   x[73][1] =  32'sd3;   x[73][2] =  32'sd890;   x[73][3] = -32'sd911;
x[74][0] =  32'sd396;   x[74][1] =  32'sd2597;   x[74][2] = -32'sd887;   x[74][3] =  32'sd1080;
x[75][0] = -32'sd810;   x[75][1] =  32'sd3;   x[75][2] =  32'sd890;   x[75][3] = -32'sd911;
x[76][0] = -32'sd913;   x[76][1] =  32'sd1783;   x[76][2] =  32'sd1064;   x[76][3] = -32'sd317;
x[77][0] =  32'sd3927;   x[77][1] = -32'sd5113;   x[77][2] = -32'sd4379;   x[77][3] =  32'sd2153;
x[78][0] = -32'sd5198;   x[78][1] =  32'sd3105;   x[78][2] =  32'sd2079;   x[78][3] = -32'sd3699;
x[79][0] = -32'sd3952;   x[79][1] =  32'sd2653;   x[79][2] =  32'sd1567;   x[79][3] =  32'sd126;
x[80][0] =  32'sd5211;   x[80][1] = -32'sd1129;   x[80][2] = -32'sd4133;   x[80][3] =  32'sd819;
x[81][0] =  32'sd536;   x[81][1] = -32'sd1484;   x[81][2] =  32'sd425;   x[81][3] =  32'sd9;
x[82][0] =  32'sd332;   x[82][1] =  32'sd2636;   x[82][2] = -32'sd525;   x[82][3] = -32'sd2547;
x[83][0] =  32'sd3075;   x[83][1] =  32'sd1106;   x[83][2] =  32'sd196;   x[83][3] =  32'sd946;
x[84][0] = -32'sd1008;   x[84][1] = -32'sd1140;   x[84][2] =  32'sd717;   x[84][3] = -32'sd1396;
x[85][0] =  32'sd769;   x[85][1] = -32'sd743;   x[85][2] = -32'sd1089;   x[85][3] = -32'sd55;
x[86][0] =  32'sd1490;   x[86][1] = -32'sd3170;   x[86][2] = -32'sd221;   x[86][3] =  32'sd3818;
x[87][0] = -32'sd1333;   x[87][1] =  32'sd1046;   x[87][2] = -32'sd4149;   x[87][3] =  32'sd1306;
x[88][0] = -32'sd913;   x[88][1] =  32'sd1783;   x[88][2] =  32'sd1064;   x[88][3] = -32'sd317;
x[89][0] = -32'sd528;   x[89][1] =  32'sd1255;   x[89][2] = -32'sd627;   x[89][3] = -32'sd443;
x[90][0] =  32'sd1360;   x[90][1] = -32'sd3765;   x[90][2] = -32'sd4962;   x[90][3] = -32'sd195;
x[91][0] =  32'sd1617;   x[91][1] = -32'sd3204;   x[91][2] = -32'sd2685;   x[91][3] =  32'sd1678;
x[92][0] =  32'sd2077;   x[92][1] = -32'sd1095;   x[92][2] =  32'sd22;   x[92][3] = -32'sd606;
x[93][0] =  32'sd2787;   x[93][1] = -32'sd314;   x[93][2] =  32'sd1752;   x[93][3] = -32'sd1212;
x[94][0] =  32'sd1296;   x[94][1] = -32'sd2099;   x[94][2] = -32'sd639;   x[94][3] =  32'sd2295;
x[95][0] =  32'sd1611;   x[95][1] =  32'sd998;   x[95][2] = -32'sd329;   x[95][3] =  32'sd765;
x[96][0] =  32'sd926;   x[96][1] = -32'sd1767;   x[96][2] =  32'sd306;   x[96][3] = -32'sd1613;
x[97][0] = -32'sd3175;   x[97][1] =  32'sd985;   x[97][2] =  32'sd5032;   x[97][3] = -32'sd2843;
x[98][0] =  32'sd509;   x[98][1] = -32'sd1189;   x[98][2] =  32'sd554;   x[98][3] = -32'sd1466;
x[99][0] = -32'sd1167;   x[99][1] =  32'sd335;   x[99][2] =  32'sd1079;   x[99][3] = -32'sd546;
x[100][0] =  32'sd1958;   x[100][1] =  32'sd80;   x[100][2] = -32'sd1962;   x[100][3] =  32'sd1445;
x[101][0] =  32'sd3971;   x[101][1] =  32'sd486;   x[101][2] = -32'sd734;   x[101][3] = -32'sd2430;
x[102][0] =  32'sd3273;   x[102][1] = -32'sd306;   x[102][2] = -32'sd2963;   x[102][3] =  32'sd2045;
x[103][0] = -32'sd592;   x[103][1] = -32'sd3;   x[103][2] =  32'sd3214;   x[103][3] = -32'sd756;
x[104][0] =  32'sd4826;   x[104][1] =  32'sd1041;   x[104][2] = -32'sd1288;   x[104][3] =  32'sd6781;
x[105][0] =  32'sd198;   x[105][1] = -32'sd2267;   x[105][2] = -32'sd2736;   x[105][3] =  32'sd4894;
x[106][0] = -32'sd1391;   x[106][1] =  32'sd3016;   x[106][2] =  32'sd2707;   x[106][3] = -32'sd3855;
x[107][0] = -32'sd3119;   x[107][1] =  32'sd4803;   x[107][2] =  32'sd1709;   x[107][3] = -32'sd1905;
x[108][0] =  32'sd4595;   x[108][1] = -32'sd1633;   x[108][2] = -32'sd3476;   x[108][3] =  32'sd3085;
x[109][0] =  32'sd933;   x[109][1] = -32'sd1018;   x[109][2] = -32'sd1417;   x[109][3] =  32'sd1864;
x[110][0] =  32'sd1887;   x[110][1] = -32'sd2719;   x[110][2] =  32'sd221;   x[110][3] = -32'sd213;
x[111][0] = -32'sd3014;   x[111][1] =  32'sd754;   x[111][2] =  32'sd2157;   x[111][3] =  32'sd434;
x[112][0] = -32'sd1457;   x[112][1] =  32'sd2048;   x[112][2] =  32'sd4077;   x[112][3] = -32'sd2397;
x[113][0] =  32'sd320;   x[113][1] = -32'sd1170;   x[113][2] =  32'sd792;   x[113][3] = -32'sd534;
x[114][0] =  32'sd5864;   x[114][1] = -32'sd979;   x[114][2] =  32'sd1751;   x[114][3] =  32'sd4077;
x[115][0] =  32'sd2749;   x[115][1] = -32'sd877;   x[115][2] = -32'sd3494;   x[115][3] = -32'sd453;
x[116][0] =  32'sd2357;   x[116][1] = -32'sd5855;   x[116][2] =  32'sd809;   x[116][3] =  32'sd7001;
x[117][0] = -32'sd7766;   x[117][1] =  32'sd2207;   x[117][2] =  32'sd7131;   x[117][3] = -32'sd4037;
x[118][0] =  32'sd3105;   x[118][1] = -32'sd2762;   x[118][2] = -32'sd430;   x[118][3] =  32'sd1700;
x[119][0] = -32'sd7766;   x[119][1] =  32'sd2207;   x[119][2] =  32'sd7131;   x[119][3] = -32'sd4037;

      
        @(posedge clk);
        mode <= 1'b1;
        
        $display("");
        $display("EXECUTANDO LSTM NETWORK");
        
        wait (ready == 1'b1);
        
        $display("");
        $display("INPUTS (em real):");
        for (t = 0; t < 5; t++) begin
            $display("  timestep[%0d]: [%.4f, %.4f, %.4f, %.4f]", 
                     t, q2real(x[t][0]), q2real(x[t][1]), q2real(x[t][2]), q2real(x[t][3]));
        end
        $display("  ... (até timestep 49)");
        $display("  timestep[48]: [%.4f, %.4f, %.4f, %.4f]", 
                 q2real(x[48][0]), q2real(x[48][1]), q2real(x[48][2]), q2real(x[48][3]));
        $display("  timestep[49]: [%.4f, %.4f, %.4f, %.4f]", 
                 q2real(x[49][0]), q2real(x[49][1]), q2real(x[49][2]), q2real(x[49][3]));
        
        // PESOS NO LSTM
        $display("");
        $display("PESOS NO LSTM:");
        $display("  lstm_wx_input[0][0] = %0d (%.6f real)", dut.lstm_inst.w_ix[0][0], q2real(dut.lstm_inst.w_ix[0][0]));
        $display("  lstm_bias_input[0] = %0d (%.6f real)", dut.lstm_inst.bias_i[0], q2real(dut.lstm_inst.bias_i[0]));
        $display("  lstm_wx_forget[0][0] = %0d (%.6f real)", dut.lstm_inst.w_fx[0][0], q2real(dut.lstm_inst.w_fx[0][0]));
        $display("  lstm_bias_forget[0] = %0d (%.6f real)", dut.lstm_inst.bias_f[0], q2real(dut.lstm_inst.bias_f[0]));
        
        // SAÍDA LSTM
        $display("");
        $display("CAMADA 1 - LSTM (último timestep):");
        for (int n = 0; n < LSTM_HIDDEN; n++) begin
            $display("  h_out[49][%0d] = %8d (%.6f real)", 
                     n, dut.lstm_inst.h_out[TIMESTEPS-1][n], 
                     q2real(dut.lstm_inst.h_out[TIMESTEPS-1][n]));
        end
        
        $display("");
        $display("  Cell state final (c_final):");
        for (int n = 0; n < LSTM_HIDDEN; n++) begin
            $display("  c_final[%0d] = %8d (%.6f real)", 
                     n, dut.lstm_inst.c_final[n], q2real(dut.lstm_inst.c_final[n]));
        end
        
        // SAÍDA ReLU
        $display("");
        $display("CAMADA 2 - ReLU:");
        for (int n = 0; n < RELU_NEURONS; n++) begin
            $display("  relu_y[%0d] = %8d (%.6f real)", 
                     n, dut.relu_inst.y[n], q2real(dut.relu_inst.y[n]));
        end
        
        // SAÍDA Sigmoid
        $display("");
        $display("CAMADA 3 - Sigmoid Output:");
        $display("  out_temp[0] = %8d (%.6f real)", 
                 dut.out_inst.y[0], q2real(dut.out_inst.y[0]));
        
        // RESULTADO FINAL
        y_frac = q2real(y_out);
        
        $display("");
        $display("RESULTADO FINAL:");
        $display("  y = %0d (Q16.16) = %.6f (real)", y_out, y_frac);
        

        
        
        $display("");
        $display("Simulação completa!");
        $finish;
    end
    
endmodule