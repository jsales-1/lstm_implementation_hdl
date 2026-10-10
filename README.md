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
