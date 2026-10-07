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
A arquitetura implementada em hardware corresponde a uma versão simplificada do modelo original de classificação binária de sentimentos. O modelo de referência em software é composto por uma camada de <i>embedding</i> de dimensão 64 com 150 tokens, duas camadas LSTM bidirecionais em cascata (64 e 32 neurônios por direção), uma camada totalmente conectada com 32 neurônios e ativação ReLU, e uma camada de saída sigmoide com 1 neurônio — totalizando 2.589.889 parâmetros treináveis e acurácia de 88,99%.
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
Devido à elevada complexidade da arquitetura original para implementação direta em hardware, foram adotadas duas versões simplificadas que preservam o funcionamento da célula LSTM, mas reduzem drasticamente o número de parâmetros e operações:
</p>

<ul>
<li>
<b>Simplified Neural Network</b> — versão intermediária, composta por uma camada de <i>embedding</i> de dimensão 4 com 120 tokens, uma camada LSTM com <b>8 neurônios</b>, uma camada ReLU com <b>8 neurônios</b> e uma saída sigmoide com <b>1 neurônio</b>. Desconsiderando-se o <i>embedding</i>, essa versão possui <b>488 parâmetros treináveis</b> e mantém uma acurácia de <b>87,53%</b> — muito próxima da arquitetura original, com uma fração dos recursos.
</li>
<li>
<b>Simplest Neural Network</b> — versão mínima, utilizada como primeiro passo de validação do fluxo RTL. Emprega <b>2 neurônios LSTM</b>, <b>4 neurônios ReLU</b> e <b>1 neurônio de saída</b>, com apenas <b>5 tokens</b> por amostra. Sua acurácia no modelo Python é de aproximadamente <b>71%</b>, o que a torna adequada para depuração rápida da infraestrutura de verificação, mas não para avaliação de desempenho.
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
Ambos os testbenches seguem a mesma estrutura de verificação, diferenciando-se apenas pelos parâmetros de configuração da rede (número de entradas, neurônios ocultos, timesteps e neurônios das camadas seguintes). Cada pasta contém ainda um arquivo <code>EDA_link.txt</code> com o link para o EDA Playground. É importante destacar que a simulação no EDA Playground <b>não contempla os 10.000 arquivos de teste</b> (<code>dados_*.mem</code>) utilizados na verificação completa — ela serve apenas como uma execução reduzida para inspeção rápida do comportamento do RTL. A verificação completa deve ser executada localmente, com todos os arquivos de estímulo disponíveis.
</p>

<h3>Fluxo de Verificação</h3>

<p align="justify">
O testbench executa as seguintes etapas em sequência:
</p>

<ol>
<li><b>Leitura dos pesos</b> (<code>weights.mem</code>) — carregados uma única vez na memória interna do testbench e escritos no DUT via interface de escrita (<code>we</code>, <code>addr</code>, <code>data_in</code>).</li>
<li><b>Iteração sobre os arquivos de teste</b> — o loop principal percorre <code>dados_0.mem</code> até <code>dados_9.mem</code> (10 arquivos), cada um representando uma amostra de entrada diferente.</li>
<li><b>Limpeza do array de entrada</b> (<code>x</code>) e <b>carregamento dos dados</b> do arquivo corrente, preenchendo a matriz <code>x[TIMESTEPS][LSTM_INPUTS]</code>.</li>
<li><b>Reset do DUT</b> e, apenas na primeira iteração, <b>escrita dos pesos</b> no banco de registradores.</li>
<li><b>Execução da rede</b> — o testbench ativa <code>mode = 1</code> e aguarda o sinal <code>ready</code>. O resultado é lido em <code>y_out</code> (Q8.24).</li>
<li><b>Comparação com o modelo Python</b> — cada arquivo <code>dados_*.mem</code> contém, além das entradas, o resultado esperado do Python e o <i>ground truth</i> (rótulo correto). O testbench converte os valores de Q8.24 para <i>real</i> e calcula acurácia, erros e a métrica R² entre Verilog e Python.</li>
<li><b>Geração de relatórios</b> — ao final, são gerados:
<ul>
<li><code>resultados.txt</code> — comparação par a par entre Verilog, Python e ground truth;</li>
<li><code>resultados_lstm.txt</code> — saída da camada LSTM (todos os neurônios ocultos, último timestep), em ponto flutuante;</li>
</ul>
</li>
</ol>

<p align="justify">
Ao final da simulação, o testbench imprime um relatório com acurácia do Python e do Verilog, contagem de acertos/erros em cada categoria (ambos corretos, ambos errados, Python errado/Verilog correto, Python correto/Verilog errado) e o coeficiente de determinação <b>R²</b> entre as duas saídas — métrica que quantifica o quão próximo o hardware está do modelo de referência.
</p>

<h2>Verificação Baseada em Asserções</h2>

<p align="justify">
O projeto conta com um conjunto de <b>asserções SystemVerilog (SVA)</b> que verificam invariantes do RTL em tempo de simulação. As asserções estão em <code>Assertions/Q8_24 V2/</code>, organizadas em duas subpastas — <code>Simplest Neural Network/</code> e <code>Simplified Neural Network/</code> — correspondentes aos dois modelos explorados. O conjunto de asserções é o mesmo para ambos; a única diferença é o <code>lstm_network</code> ao qual são vinculadas via <code>bind</code>.
</p>

<p align="justify">
O arquivo <code>lstm_assertions.sv</code> verifica o comportamento de reset e clear (garantindo que <code>ready</code> e <code>y_out</code> são zerados corretamente), a sequência do sinal <code>ready</code> (que só pode estar alto em <code>mode = 1</code>, por tempo limitado, e nunca em modo de escrita), os modos de operação (assegurando que <code>we</code> nunca é ativado durante a execução), a validade da saída (quando <code>ready = 1</code>, <code>y_out</code> está em <code>[0, 1]</code> em Q8.24) e a integridade das FSMs do <code>lstm_network</code> (cujo estado deve estar sempre entre os 36 estados válidos de carregamento, execução, espera e conclusão) e do <code>lstm_layer</code> (cujo estado deve ser sempre <code>IDLE</code>, <code>COMPUTE</code> ou <code>DONE</code>). Complementam a verificação cinco covergroups — <code>cg_modes</code>, <code>cg_reset</code>, <code>cg_ready</code>, <code>cg_yout</code> e <code>cg_lstm_state</code> — que registram a ocorrência dos cenários relevantes de operação e as faixas de saída alcançadas durante a simulação.
</p>

<h2>Comparação entre Python e SystemVerilog</h2>

<p align="justify">
O diretório <b>Comparing Python and SystemVerilog</b> contém os resultados da comparação entre os modelos implementados em Python e SystemVerilog. Esta comparação permite validar a precisão numérica das operações em ponto fixo e garantir que o hardware produza resultados consistentes com o modelo de referência. Os resultados são organizados por formato numérico (Q8.24 e Q16.16), permitindo avaliar o impacto da escolha do formato na acurácia final da rede.
</p>
