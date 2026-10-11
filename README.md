<h1>Implementação e Verificação de Rede Neural LSTM em Linguagem de Descrição de Hardware</h1>

<p align="justify">
Este repositório contém o projeto de implementação e verificação de uma rede neural LSTM (Long Short-Term Memory) em hardware, desenvolvido para a disciplina SD292 - Trabalho Orientado II. O projeto abrange desde a descrição RTL em SystemVerilog até a verificação funcional utilizando testbenches direcionados e asserções.
</p>

<h2>Estrutura do Projeto</h2>

<pre>
LSTM_IMPLEMENTATION_HDL/
├── Assertions/                                 # Asserções SystemVerilog (SVA)
│   └── Q8_24 V2/                               # Asserções para Q8.24 V2
│       ├── Simplest Neural Network/            # Testes com a rede mais simples
│       └── Simplified Neural Network/          # Testes com a rede simplificada
├── Comparing Python and SystemVerilog/         # Comparação entre modelos
│   └── Q8_24/                                  # Comparação em Q8.24
├── Geral Tests SystemVerilog/                  # Testbenches gerais em SystemVerilog
│   ├── Q16_16/                                 # Testes em Q16.16
│   │   ├── Simplest Neural Network/            # Rede neural mais simples
│   │   ├── Simplest Neural Network 2.0/        # Versão 2.0 da rede mais simples
│   │   ├── Simplified Neural Network/          # Rede neural simplificada
│   │   └── Simplified Neural Network 2.0/      # Versão 2.0 da rede simplificada
│   ├── Q8_24/                                  # Testes em Q8.24
│   │   ├── Simplest Neural Network/            # Rede neural mais simples
│   │   └── Simplified Neural Network/          # Rede neural simplificada
│   └── Q8_24 V2/                               # Testes em Q8.24 (versão 2)
│       ├── Simplest Neural Network/            # Rede neural mais simples
│       │   └── Small Global Processing/        # Processamento global (dados_*.mem)
│       └── Simplified Neural Network/          # Rede neural simplificada
│           └── Global Processing/              # Processamento global (dados_*.mem)
├── Neuralnetworks Initial Tests/               # Testes iniciais das redes neurais
│   ├── Creating_Perceptrons/                   # Implementação de perceptrons
│   ├── EDA_link.txt                            # Link para EDA Playground
│   ├── regbank_addr21.sv                       # Banco de registradores (addr21)
│   ├── simples_nn.sv                           # Rede neural simples
│   └── testbench.sv                            # Testbench inicial
├── Training NN's/                              # Treinamento das redes neurais
│   ├── Initial Tests/                          # Testes iniciais de treinamento
│   └── LSTM Final Model/                       # Modelo LSTM final treinado
│       ├── Modo Q16_16/                        # Modelo em Q16.16
│       ├── Modo Q8_24/                         # Modelo em Q8.24
│       │   └── Global Processing/              # Processamento global (dados_*.mem)
│       └── Modo Q8_24 Version 2.0/             # Modelo em Q8.24 (versão 2.0)
│           └── Global Processing/              # Processamento global (dados_*.mem)
├── Verification Plan and Specification/        # Plano e especificação de verificação
│   └── Project Verification - José David.docx  # Documento de verificação
│   └── Project Specification - José David.docx # Documento de verificação
├── Synthesis/                                  # Pasta de sínteses 
│   └── Creating Netlists/                      # Criação de netlists a partir dos RTLs
│   └── Gate Level Simulation/                  # Simulação com as netlist geradas
</pre>

<h2>Register Transfer Level (RTL)</h2>

<p align="justify">
O projeto é composto pelos seguintes módulos principais:
</p>

<ul>
<li><b>lstm_network</b>: Módulo top-level que integra todas as camadas da rede e controla o fluxo de execução.</li>
<li><b>lstm_layer</b>: Implementa a camada LSTM completa com FSM para controle temporal.</li>
<li><b>lstm_cell</b>: Representa uma camada LSTM com múltiplos neurônios operando em paralelo.</li>
<li><b>lstm_cell_neuron</b>: Implementa um único neurônio LSTM com portas de esquecimento, entrada, candidato e saída.</li>
<li><b>relu_layer</b>: Camada totalmente conectada com ativação ReLU.</li>
<li><b>sigmoid_layer</b>: Camada de saída com ativação sigmoide.</li>
<li><b>weight_bank</b>: Banco de registradores para armazenamento de pesos e biases.</li>
<li><b>mac, sigmoid, tanh</b>: Módulos auxiliares para operações aritméticas e funções de ativação.</li>
</ul>

<p align="justify">
A arquitetura implementada em hardware corresponde a uma versão simplificada do modelo original de classificação binária de sentimentos. O modelo de referência em software é composto por uma camada de <i>embedding</i> de dimensão 64 com 150 tokens, duas camadas LSTM bidirecionais em cascata (64 e 32 neurônios por direção), uma camada totalmente conectada com 32 neurônios e ativação ReLU, e uma camada de saída sigmoide com 1 neurônio, totalizando 2.589.889 parâmetros treináveis e acurácia de 88,99%.
</p>

<p align="justify">
A versão implementada em RTL suprime a camada de <i>embedding</i> (a rede recebe diretamente o vetor de características) e adota uma única camada LSTM com <b>8 neurônios ocultos</b>, uma camada ReLU com <b>8 neurônios</b> e uma saída sigmoide com <b>1 neurônio</b>. A entrada é uma matriz de <b>120 passos temporais × 4 características</b>. Desconsiderando o <i>embedding</i>, a versão em hardware possui <b>488 parâmetros treináveis</b> e mantém acurácia de aproximadamente <b>87,53%</b> no modelo Python equivalente. Todas as operações aritméticas utilizam o formato de ponto fixo Q8.24 (8 bits para parte inteira e 24 bits para parte fracionária).
</p>

<h2>Testbenches Direcionados</h2>

<p align="justify">
A verificação funcional do projeto foi conduzida por meio de <b>testbenches direcionados</b>, que comparam a saída do hardware em SystemVerilog com o resultado de referência obtido pelo modelo equivalente implementado em Python. Para que essa comparação fosse viável dentro das restrições de recursos do projeto, foram explorados <b>dois modelos de rede</b> com níveis distintos de complexidade, ambos derivados da arquitetura original.
</p>

<h3>Modelos Explorados</h3>

<p align="justify">
Devido à complexidade da arquitetura original para implementação direta em hardware, foram adotadas duas versões simplificadas que preservam o funcionamento da célula LSTM, mas reduzem o número de parâmetros e operações:
</p>

<ul>
<li>
<b>Simplified Neural Network</b>: versão intermediária, composta por uma camada de <i>embedding</i> de dimensão 4 com 120 tokens, uma camada LSTM com <b>8 neurônios</b>, uma camada ReLU com <b>8 neurônios</b> e uma saída sigmoide com <b>1 neurônio</b>. Desconsiderando-se o <i>embedding</i>, essa versão possui <b>488 parâmetros treináveis</b> e mantém uma acurácia de <b>87,53%</b>, próxima da arquitetura original, com uma fração dos recursos.
</li>
<li>
<b>Simplest Neural Network</b>: versão mínima, utilizada como primeiro passo de validação do fluxo RTL. Emprega <b>2 neurônios LSTM</b>, <b>4 neurônios ReLU</b> e <b>1 neurônio de saída</b>, com apenas <b>5 tokens</b> por amostra. Sua acurácia no modelo Python é de aproximadamente <b>71%</b>, o que a torna adequada para depuração rápida da infraestrutura de verificação, mas não para avaliação de desempenho.
</li>
</ul>

<p align="justify">
Essa exploração em dois níveis permitiu validar incrementalmente o fluxo de verificação: primeiro com uma rede pequena o suficiente para inspeção manual dos resultados, e depois com uma rede que se aproxima do comportamento do modelo original em software.
</p>

<h3>Estrutura dos Testbenches</h3>

<p align="justify">
Cada modelo possui seu próprio testbench direcionado, localizado em:
</p>

<ul>
<li><code>Geral Tests SystemVerilog/Q8_24 V2/Simplest Neural Network/testbench.sv</code></li>
<li><code>Geral Tests SystemVerilog/Q8_24 V2/Simplified Neural Network/testbench.sv</code></li>
</ul>

<p align="justify">
Ambos os testbenches seguem a mesma estrutura de verificação, diferenciando-se apenas pelos parâmetros de configuração da rede (número de entradas, neurônios ocultos, timesteps e neurônios das camadas seguintes). Cada pasta contém ainda um arquivo <code>EDA_link.txt</code> com o link para o EDA Playground. A simulação no EDA Playground <b>não contempla os 10.000 arquivos de teste</b> (<code>dados_*.mem</code>) utilizados na verificação completa, servindo apenas como uma execução reduzida para inspeção rápida do comportamento do RTL. A verificação completa deve ser executada localmente, com todos os arquivos de estímulo disponíveis.
</p>

<h3>Fluxo de Verificação</h3>

<p align="justify">
O testbench executa a verificação em etapas sequenciais. Inicialmente, os pesos são lidos do arquivo <code>weights.mem</code> e carregados uma única vez na memória interna do testbench, sendo em seguida escritos no DUT por meio da interface de escrita (<code>we</code>, <code>addr</code> e <code>data_in</code>). Na sequência, o loop principal percorre os arquivos de teste, cada um representando uma amostra de entrada diferente. Para cada arquivo, o array de entrada <code>x</code> é limpo e preenchido com os dados do arquivo corrente, formando a matriz <code>x[TIMESTEPS][LSTM_INPUTS]</code>. O DUT é então resetado e, apenas na primeira iteração, os pesos são efetivamente escritos no banco de registradores. Em seguida, o testbench ativa <code>mode = 1</code> e aguarda o sinal <code>ready</code>, lendo o resultado em <code>y_out</code> no formato Q8.24. Esse resultado é convertido para ponto flutuante e comparado com o valor esperado do modelo Python, que está armazenado no próprio arquivo de teste junto com o <i>ground truth</i> (rótulo correto). A partir dessa comparação, o testbench calcula acurácia, contagem de erros e a métrica R² entre Verilog e Python. Ao final, são gerados dois arquivos de relatório: <code>resultados.txt</code>, com a comparação par a par entre Verilog, Python e ground truth, e <code>resultados_lstm.txt</code>, com a saída da camada LSTM (todos os neurônios ocultos no último timestep) em ponto flutuante. Por fim, o testbench imprime um relatório com a acurácia de cada implementação, a contagem de acertos e erros em cada categoria (ambos corretos, ambos errados, Python errado com Verilog correto e Python correto com Verilog errado) e o coeficiente de determinação <b>R²</b> entre as duas saídas, métrica que quantifica o quão próximo o hardware está do modelo de referência.
</p>

<h2>Verificação Baseada em Asserções</h2>

<p align="justify">
O projeto conta com um conjunto de <b>asserções SystemVerilog (SVA)</b> que verificam invariantes do RTL em tempo de simulação. As asserções estão em <code>Assertions/Q8_24 V2/</code>, organizadas em duas subpastas, <code>Simplest Neural Network/</code> e <code>Simplified Neural Network/</code>, correspondentes aos dois modelos explorados. O conjunto de asserções é o mesmo para ambos; a única diferença é o <code>lstm_network</code> ao qual são vinculadas via <code>bind</code>.
</p>

<p align="justify">
O arquivo <code>lstm_assertions.sv</code> verifica o comportamento de reset e clear (garantindo que <code>ready</code> e <code>y_out</code> são zerados corretamente), a sequência do sinal <code>ready</code> (que só pode estar alto em <code>mode = 1</code>, por tempo limitado, e nunca em modo de escrita), os modos de operação (assegurando que <code>we</code> nunca é ativado durante a execução), a validade da saída (quando <code>ready = 1</code>, <code>y_out</code> está em <code>[0, 1]</code> em Q8.24) e a integridade das FSMs do <code>lstm_network</code> (cujo estado deve estar sempre entre os 36 estados válidos de carregamento, execução, espera e conclusão) e do <code>lstm_layer</code> (cujo estado deve ser sempre <code>IDLE</code>, <code>COMPUTE</code> ou <code>DONE</code>). Complementam a verificação cinco covergroups, <code>cg_modes</code>, <code>cg_reset</code>, <code>cg_ready</code>, <code>cg_yout</code> e <code>cg_lstm_state</code>, que registram a ocorrência dos cenários relevantes de operação e as faixas de saída alcançadas durante a simulação.
</p>

<h2>Relatórios de Cobertura</h2>

<p align="justify">
Os resultados de cobertura funcional foram gerados pelo Cadence IMC (Incisive Metrics Center) e estão disponíveis em formato HTML dentro das pastas de cada modelo. Para o <b>Simplest Neural Network</b>, os relatórios estão em <code>Geral Tests SystemVerilog/Q8_24 V2/Simplest Neural Network/html_imc_simplest</code>; para o <b>Simplified Neural Network</b>, em <code>Geral Tests SystemVerilog/Q8_24 V2/Simplified Neural Network/html_imc_simplified</code>. Para visualizar os relatórios, basta abrir o arquivo <code>index.html</code> de cada pasta em um navegador. 
</p>


<h2>Comparação entre Python e SystemVerilog</h2>

<p align="justify">
O diretório <b>Comparing Python and SystemVerilog</b> contém os resultados da comparação entre os modelos implementados em Python e SystemVerilog. Esta comparação permite validar a precisão numérica das operações em ponto fixo e garantir que o hardware produza resultados consistentes com o modelo de referência. Os resultados são organizados por formato numérico (Q8.24 e Q16.16) e, dentro de cada formato, pelos dois modelos explorados (Simplest e Simplified).
</p>

<h3>Simplest Neural Network (Q8.24)</h3>

<p align="justify">
O modelo <b>Simplest</b>, composto por 5 tokens, 2 neurônios LSTM, 4 neurônios ReLU e 1 neurônio sigmoide, foi o primeiro a ser comparado. A matriz de confusão abaixo apresenta os resultados lado a lado entre Verilog e Python, ambos com acurácia de aproximadamente 71%:
</p>

<p align="center">
<img src="./Comparing%20Python%20and%20SystemVerilog/Q8_24/comparacao_modelo_simplest_cm.png" alt="Matriz de confusão Simplest" width="800">
</p>

<p align="justify">
A matriz mostra que o número de acertos e erros é praticamente idêntico entre as duas implementações. O Verilog classificou corretamente 3266 amostras como "Negative" e 3837 como "Positive", enquanto o Python obteve 3268 e 3839, respectivamente. As diferenças são de apenas 2 amostras em cada classe, o que indica alta concordância entre o hardware e o modelo de referência.
</p>

<p align="justify">
Os histogramas das saídas permitem observar a distribuição das probabilidades geradas por cada implementação:
</p>

<p align="center">
<img src="./Comparing%20Python%20and%20SystemVerilog/Q8_24/comparacao_modelo_simplest_histograms.png" alt="Histograma Simplest" width="800">
</p>

<p align="justify">
Ambas as distribuições apresentam o mesmo padrão: uma concentração de amostras próximas de 0 (classificadas como "Negative") e outra próxima de 0,7 (classificadas como "Positive"), com poucos valores na região intermediária. O formato geral dos histogramas é equivalente, confirmando que o hardware reproduz o comportamento do modelo Python. Nota-se ainda um pequeno acúmulo de amostras em torno de 0,5. Esse comportamento é esperado e decorre da saturação da função sigmoide implementada em hardware: para entradas muito negativas, a aproximação polinomial por série de Taylor combinada ao mecanismo de clamp produz valores em torno de 0,5, enquanto no modelo Python esses mesmos valores tenderiam a zero.
</p>

<p align="justify">
Por fim, o gráfico de dispersão abaixo relaciona diretamente as saídas de Python (eixo X) e Verilog (eixo Y):
</p>

<p align="center">
<img src="./Comparing%20Python%20and%20SystemVerilog/Q8_24/comparacao_modelo_simplest_r2_scatter.png" alt="Scatter plot Simplest" width="700">
</p>

<p align="justify">
A linha tracejada vermelha representa a correspondência perfeita (y = x). A maior parte dos pontos está alinhada a essa linha, resultando em um coeficiente de determinação <b>R² = 0,9385</b>, valor que indica alta correlação entre as saídas. Os desvios mais visíveis ocorrem em valores baixos de saída, onde a quantização em Q8.24 e as aproximações polinomiais das funções de ativação introduzem pequenos erros. Ainda assim, esses desvios não comprometem a classificação final, como mostrado pela matriz de confusão.
</p>

<p align="justify">
O plano de verificação do projeto define <b>R² ≥ 0,9</b> como critério de aceitação para a concordância entre o hardware e o modelo de referência. O valor obtido para o modelo Simplest em Q8.24 (<b>R² = 0,9385</b>) atende a esse critério, confirmando que a implementação em hardware produz resultados equivalentes ao modelo em Python, tanto em termos de acurácia final quanto na distribuição das probabilidades geradas.
</p>

<h3>Simplified Neural Network (Q8.24)</h3>

<p align="justify">
O modelo <b>Simplified</b>, composto por 120 tokens, 8 neurônios LSTM, 8 neurônios ReLU e 1 neurônio sigmoide, é a versão que mais se aproxima do comportamento do modelo original em software. A matriz de confusão abaixo apresenta os resultados lado a lado entre Verilog e Python, com acurácia de 87,56% e 87,53%, respectivamente:
</p>

<p align="center">
<img src="./Comparing%20Python%20and%20SystemVerilog/Q8_24/comparacao_modelo_simplified_cm.png" alt="Matriz de confusão Simplified" width="800">
</p>

<p align="justify">
A matriz mostra que as duas implementações produziram resultados quase idênticos. O Verilog classificou corretamente 4103 amostras como "Negative" e 4653 como "Positive", enquanto o Python obteve 4103 e 4650, respectivamente. As diferenças são de apenas 3 amostras na classe "Positive", o que confirma a alta concordância entre hardware e modelo de referência mesmo em uma rede maior.
</p>

<p align="justify">
Os histogramas das saídas permitem observar a distribuição das probabilidades geradas por cada implementação:
</p>

<p align="center">
<img src="./Comparing%20Python%20and%20SystemVerilog/Q8_24/comparacao_modelo_simplified_histograms.png" alt="Histograma Simplified" width="800">
</p>

<p align="justify">
Ambas as distribuições concentram a maior parte das amostras próximas de 0 (classificadas como "Negative") e de 1,0 (classificadas como "Positive"), com poucos valores na região intermediária. O comportamento é semelhante ao observado no modelo Simplest, incluindo o acúmulo em torno de 0,5 que, novamente, decorre da saturação da função sigmoide em hardware. A principal diferença em relação ao Simplest é que, no Simplified, as saídas positivas saturam em 1,0 (em vez de 0,7), refletindo a maior confiança do modelo com mais neurônios e mais tokens.
</p>

<p align="justify">
Por fim, o gráfico de dispersão abaixo relaciona diretamente as saídas de Python (eixo X) e Verilog (eixo Y):
</p>

<p align="center">
<img src="./Comparing%20Python%20and%20SystemVerilog/Q8_24/comparacao_modelo_simplified_r2_scatter.png" alt="Scatter plot Simplified" width="700">
</p>

<p align="justify">
A linha tracejada vermelha representa a correspondência perfeita (y = x). A maior parte dos pontos está alinhada a essa linha, resultando em um coeficiente de determinação <b>R² = 0,9023</b>. Os desvios mais visíveis ocorrem em valores baixos de saída e em dois agrupamentos verticais: um em torno de Python ≈ 0,1 (onde o Verilog produz valores entre 0,35 e 0,5) e outro em torno de Python ≈ 0,95 (onde o Verilog produz valores entre 0,5 e 0,65). Esses agrupamentos são consequência direta da saturação da sigmoide e do truncamento em Q8.24 para entradas muito negativas ou muito positivas.
</p>

<p align="justify">
Apesar desses desvios, o valor obtido para o modelo Simplified em Q8.24 (<b>R² = 0,9023</b>) atende ao critério de aceitação definido no plano de verificação (<b>R² ≥ 0,9</b>), confirmando que mesmo a rede de maior porte mantém resultados equivalentes ao modelo Python.
</p>
<h2>Síntese e Relatórios</h2>

<p align="justify">
O processo de síntese foi realizado utilizando o Cadence Genus, com os scripts de síntese disponíveis no diretório <code>Synthesis/Creating_Netlists/script/</code>. Os arquivos <code>lstm_simplest.tcl</code> e <code>lstm_simplified.tcl</code> contêm a configuração completa de síntese para os modelos Simplest e Simplified, respectivamente, e podem ser executados no Genus por meio do comando <code>source</code>. Para a visualização gráfica das netlists geradas, os scripts <code>gui_show_simplest.tcl</code> e <code>gui_show_simplified.tcl</code> abrem a interface gráfica do Genus com a netlist correspondente já carregada. As restrições utilizadas durante a síntese, incluindo a definição de clock e os atrasos de entrada e saída, estão disponíveis no arquivo de constraints acessível em <code>Synthesis/Creating_Netlists/constraints/</code>. A tecnologia alvo foi a biblioteca de células de 45 nm da Nangate.
</p>

<p align="justify">
Os relatórios de síntese resultantes estão organizados nas subpastas <code>report_simplest</code> e <code>report_simplified</code>, também localizadas sob o diretório <code>script/</code>. Cada pasta contém os relatórios gerados para área (<code>report_area.rpt</code>), potência (<code>report_power.rpt</code>), timing (<code>report_timing.rpt</code>) e qualidade geral dos resultados (<code>report_qor.rpt</code>). Esses relatórios fornecem uma análise detalhada de Power, Performance e Area (PPA) do design sintetizado, permitindo avaliar a viabilidade da implementação e orientar possíveis otimizações no RTL antes de prosseguir para o projeto físico.
</p>

<p align="justify">
O fluxo de síntese depende de arquivos de tecnologia proprietários e bibliotecas de células padrão que não estão incluídos neste repositório. Para executar com sucesso os scripts TCL fornecidos, o usuário deve ter acesso às bibliotecas compatíveis com o Cadence, incluindo as bibliotecas de células padrão <code>fast_vdd1v0_basicCells</code> e <code>slow_vdd1v0_basicCells</code> (ambos os arquivos <code>.lib</code> e os correspondentes <code>.v</code> em Verilog), bem como os arquivos de tecnologia LEF <code>gsclib045_macro.lef</code> e <code>gsclib045_tech.lef</code>. Esses arquivos devem ser obtidos separadamente (por exemplo, a partir de uma distribuição acadêmica ou licenciada, como a biblioteca Nangate 45 nm) e colocados manualmente nos diretórios apropriados sob <code>Synthesis/Creating_Netlists/</code>. Devido a restrições de licenciamento, esses recursos não podem ser redistribuídos como parte deste repositório.
</p>

<h3>Análise Geral dos Resultados</h3>

<p align="justify">
Os dois modelos foram sintetizados com sucesso para um período de clock de 10 ns, o que corresponde a uma frequência de operação de 100 MHz, sem violações de setup em nenhum dos casos. Tanto o Simplest quanto o Simplified apresentaram slack de 0,0 ps e TNS de 0,0, o que indica que o caminho crítico fecha exatamente no limite do período especificado, sem margem positiva. O caminho crítico em ambos os casos está inteiramente dentro de um único neurônio LSTM, passando pelas etapas de MAC, árvores de soma CSA, multiplicadores e blocos de ativação sigmoide e tanh. Como a estrutura interna do neurônio é a mesma nos dois modelos, o caminho crítico não cresce com o aumento do número de neurônios ou de timesteps, o que explica por que o Simplified, com quatro vezes mais neurônios LSTM e 24 vezes mais timesteps, fecha no mesmo período do Simplest. Isso mostra que a profundidade lógica do neurônio é o fator determinante do timing, e não o tamanho total da rede.
</p>

<p align="justify">
Em termos de área, o Simplified ocupa aproximadamente 6,38 mm², cerca de 2,94 vezes a área do Simplest, que fica em torno de 2,17 mm². Esse crescimento vem de três fatores combinados: o número de neurônios LSTM passou de 2 para 8, o número de entradas de cada MAC passou de 6 para 12 por causa do vetor de entrada maior, e o número de timesteps subiu de 5 para 120. O lstm_inst concentra a maior parte da área nos dois modelos, com 36,9 por cento no Simplest e 72,3 por cento no Simplified, enquanto o weight_bank permanece praticamente constante, em torno de 1,27 mm², porque seu tamanho é fixo pela configuração MAX_SIZE4096. A relu_inst e a out_inst têm participação pequena em ambos os casos, somando menos de 5 por cento da área total.
</p>

<p align="justify">
Em potência, o Simplified consome aproximadamente 453,4 mW, cerca de três vezes o consumo do Simplest, que fica em torno de 150,3 mW. O leakage é baixo nos dois modelos, representando 0,17 por cento do total, o que é compatível com a biblioteca de 1,32 V e 0 °C. A diferença mais relevante está na distribuição entre registradores e lógica combinacional: no Simplest, os registradores respondem por 41,66 por cento do consumo e a lógica por 58,34 por cento, enquanto no Simplified os registradores caem para 15,35 por cento e a lógica sobe para 84,65 por cento. Esse contraste mostra que o aumento do tamanho da rede se traduz quase inteiramente em mais lógica combinacional, e não em mais registradores, o que é coerente com a estrutura da LSTM, onde o estado recorrente é armazenado em um conjunto relativamente pequeno de registradores enquanto as operações de multiplicação e soma se expandem em árvores combinacionais profundas. Comparando a contagem de instâncias, o número de instâncias sequenciais cresceu apenas 10,6 por cento, de 133.661 para 147.805, enquanto o número de instâncias combinacionais cresceu 300,3 por cento, de 661.446 para 2.647.988.
</p>

<p align="justify">
De forma geral, os resultados de síntese mostram que a implementação é viável em 100 MHz para os dois modelos, com o Simplified mantendo a mesma frequência do Simplest apesar do aumento expressivo de complexidade. O custo desse crescimento aparece principalmente em área e potência, que escalam de forma aproximadamente proporcional ao número de neurônios e ao número de timesteps, enquanto o timing permanece estável porque o caminho crítico está confinado à estrutura interna de um único neurônio. Essa característica é relevante para o projeto, porque indica que otimizações de timing devem focar na profundidade lógica do neurônio, e não no tamanho da rede, enquanto otimizações de área e potência podem se beneficiar de técnicas como compartilhamento de multiplicadores, redução de precisão ou clock gating.
</p>

<h2>Gate Level Simulation (GLS)</h2>

<p align="justify">
Após a síntese, foi realizada a simulação em nível de porta (Gate Level Simulation, GLS) para validar a netlist gerada contra o comportamento do RTL original. Os arquivos de GLS estão organizados no diretório <code>Synthesis/Gate Level Simulation/</code>, que contém duas subpastas, <code>simplest/</code> e <code>simplified/</code>, correspondentes aos dois modelos explorados. A simulação foi conduzida com a netlist sintetizada pelo Cadence Genus e os mesmos estímulos utilizados na verificação funcional em SystemVerilog, permitindo comparar diretamente a saída do hardware em nível de porta com a saída do RTL.
</p>

<p align="justify">
Para o modelo <b>Simplest</b>, o GLS foi realizado de forma completa, com os 10.000 exemplos de teste, e os resultados obtidos foram idênticos aos do modelo em SystemVerilog. Isso confirma que a síntese preservou a funcionalidade do RTL, apesar do aumento expressivo no número de células e da introdução de atrasos de propagação inerentes à netlist. O gráfico abaixo relaciona as saídas do Verilog (eixo X) e do SystemVerilog (eixo Y) para o modelo Simplest, mostrando a correspondência exata entre as duas implementações:
</p>

<p align="center">
<img src="./Comparing%20Python%20and%20SystemVerilog/Q8_24/comparacao_modelo_simplest_gls_sv.png" alt="GLS Simplest: SystemVerilog vs Verilog" width="700">
</p>

<p align="justify">
A linha tracejada vermelha representa a correspondência perfeita (y = x) e o coeficiente de determinação obtido foi <b>R² = 1,000000</b>, valor que indica que o hardware em nível de porta reproduz exatamente o comportamento do RTL. Esse resultado valida a etapa de síntese do modelo Simplest e confirma que a netlist gerada é funcionalmente equivalente ao design original.
</p>

<p align="justify">
Para o modelo <b>Simplified</b>, não foi possível realizar o teste completo com os 10.000 exemplos, porque a simulação em nível de porta se tornou proibitiva em tempo de execução devido ao grande número de células da netlist, que ultrapassa 2,7 milhões de instâncias. Ainda assim, os exemplos que foram testados apresentaram resultados consistentes com o RTL, indicando que a síntese também preservou a funcionalidade do modelo Simplified. Essa limitação é esperada em projetos de maior porte e não compromete a validação da etapa de síntese, já que os testes parciais cobriram uma variedade representativa de amostras.
</p>
