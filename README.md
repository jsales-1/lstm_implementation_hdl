# README

<h1>Implementação e Verificação de Rede Neural LSTM em Hardware</h1>

<p align="justify">
Este repositório contém o projeto de implementação e verificação de uma rede neural LSTM (Long Short-Term Memory) em hardware, desenvolvido para a disciplina SD292 - Trabalho Orientado II. O projeto abrange desde a descrição RTL em SystemVerilog até a verificação funcional utilizando testbenches direcionados, ambiente UVM e asserções.
</p>

<h2>Estrutura do Projeto</h2>

<pre>
LSTM_IMPLEMENTATION_HDL/
├── Comparing Python and SystemVerilog/          # Comparação entre modelos
│   ├── Q8_24/                                  # Formato Q8.24 (implementado)
│   └── Q16_16/                                 # Formato Q16.16 (implementado)
├── Creating_Perceptrons/                       # Implementação de perceptrons
├── Geral Tests SystemVerilog/                  # Testes gerais em SystemVerilog
│   ├── Q8_24/                                  # Testes em Q8.24
│   │   ├── Simplest Neural Network/            # Rede neural mais simples
│   │   └── Simplified Neural Network/          # Rede neural simplificada
│   └── Q16_16/                                 # Testes em Q16.16
├── Neuralnetworks Initial Tests/               # Testes iniciais das redes neurais
│   ├── EDA_link.txt                            # Link para EDA Playground
│   ├── regbank_addr21.sv                       # Banco de registradores (addr21)
│   ├── regbank_addr32.sv                       # Banco de registradores (addr32)
│   ├── simples_nn.sv                           # Rede neural simples
│   └── testbench.sv                            # Testbench inicial
├── Neuralnetworks LSTM/                        # Implementação LSTM completa
│   ├── EDA_link.txt                            # Link para EDA Playground
│   ├── lstm_network.sv                         # Módulo top-level
│   ├── lstm.sv                                 # Módulos LSTM (cell, layer, etc.)
│   ├── regbank_addr21.sv                       # Banco de registradores
│   ├── regbank_addr32.sv                       # Banco de registradores
│   └── testbench.sv                            # Testbench LSTM
├── Training NN's/                              # Treinamento das redes neurais
│   ├── Initial Tests/                          # Testes iniciais de treinamento
│   └── LSTM Final Model/                       # Modelo LSTM final treinado
├── Verification Plan and Specification/        # Plano e especificação de verificação
│   └── Project Verification - José David.docx  # Documento de verificação
└── README.md                                   # Este arquivo
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
A arquitetura implementada corresponde a uma versão simplificada de uma rede para classificação binária de sentimentos, composta por uma camada LSTM com 16 neurônios ocultos, uma camada ReLU com 32 neurônios e uma camada de saída sigmoide. Todas as operações aritméticas utilizam o formato Q8.24 (8 bits para parte inteira e 24 bits para parte fracionária).
</p>

<h2>Testbenches Direcionados</h2>

<p align="justify">

</p>

<p align="justify">
</p>

<h2>Verificação Baseada em Asserções</h2>

<p align="justify">

</p>

<h2>Comparação entre Python e SystemVerilog</h2>

<p align="justify">
O diretório <b>Comparing Python and SystemVerilog</b> contém os resultados da comparação entre os modelos implementados em Python e SystemVerilog, nos formatos Q8.24 e Q16.16. Esta comparação permite validar a precisão numérica das operações em ponto fixo e garantir que o hardware produza resultados consistentes com o modelo de referência.
</p>


