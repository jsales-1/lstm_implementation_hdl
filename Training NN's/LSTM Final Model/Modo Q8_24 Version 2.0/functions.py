import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SimpleRNN, Input, SpatialDropout1D
import re
import random
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def preprocess(text):
    text = re.sub(r'\\', '', text)
    text = contractions.fix(text)

    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\d\W]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [t for t in tokens if len(t) >= 3]

    return " ".join(tokens)


# ============================================================
# MUDANÇA PARA Q8.24 (FRAC = 24)
# ============================================================
FRAC = 24
SCALE = 1 << FRAC  # 2^24 = 16777216

INT32_MAX = np.int32(2**31 - 1)
INT32_MIN = np.int32(-(2**31))

def float_to_q(val):
    """Converte float para Q8.24 (formato signed 32 bits)"""
    if np.isnan(val):
        return np.int32(0)
    if np.isposinf(val):
        return INT32_MAX
    if np.isneginf(val):
        return INT32_MIN
    fixed = np.round(val * SCALE)
    return np.int32(fixed)


# ============================================================
# ENDERECO DE 12 BITS (NOVO FORMATO)
# ============================================================
# Bits 11-10: layer     (2 bits) → 0=LSTM, 1=ReLU, 2=Output
# Bit  9:     is_bias   (1 bit)  → 0=peso, 1=bias
# Bits 8-7:   gate      (2 bits) → 00=input, 01=forget, 10=candidate, 11=output
# Bits 6-4:   neuron    (3 bits) → 0 a 7
# Bit  3:     recurrent (1 bit)  → 0=Wx, 1=Wh
# Bits 2-0:   idx       (3 bits) → 0 a 7

def build_addr_12bit(layer, is_bias, gate, neuron, recurrent, idx):
    """
    Constrói endereço de 12 bits no novo formato
    
    Args:
        layer: 2 bits (0-3)
        is_bias: 1 bit (0=peso, 1=bias)
        gate: 2 bits (0-3) - apenas para LSTM
        neuron: 3 bits (0-7)
        recurrent: 1 bit (0=Wx, 1=Wh) - apenas para LSTM
        idx: 3 bits (0-7)
    """
    addr = 0
    addr |= (layer     & 0x3) << 10  # bits 11-10: layer
    addr |= (is_bias   & 0x1) << 9   # bit 9:     bias
    addr |= (gate      & 0x3) << 7   # bits 8-7:  gate
    addr |= (neuron    & 0x7) << 4   # bits 6-4:  neuron
    addr |= (recurrent & 0x1) << 3   # bit 3:     recurrent
    addr |= (idx       & 0x7)       # bits 2-0:  idx
    return addr


# ============================================================
# EXPORTA PESOS DA LSTM (NOVO FORMATO 12 BITS)
# ============================================================
def export_lstm_weights(model, output_file="weights.mem", max_neurons=8, max_idx=8):
    """
    Exporta pesos da LSTM no novo formato de 12 bits
    
    Args:
        model: Modelo Keras
        output_file: Nome do arquivo de saída
        max_neurons: Número máximo de neurônios (padrão 8)
        max_idx: Número máximo de índices (padrão 8)
    """
    print("\n" + "="*50)
    print("MODO LSTM: Exportando pesos para 12 bits")
    print("Ignorando: Embedding e SpatialDropout1D")
    print("="*50)
    
    mem = {}
    layer_idx = 0
    
    for layer in model.layers:
        if len(layer.get_weights()) == 0:
            continue
            
        if isinstance(layer, Bidirectional):
            print(f"\nCamada {layer_idx}: Bidirectional LSTM")
            
            forward_lstm = layer.forward_layer
            backward_lstm = layer.backward_layer
            
            export_lstm_core_12bit(forward_lstm, mem, layer_idx, direction=0, 
                                   max_neurons=max_neurons, max_idx=max_idx)
            export_lstm_core_12bit(backward_lstm, mem, layer_idx + 1, direction=1,
                                   max_neurons=max_neurons, max_idx=max_idx)
            
            layer_idx += 2
            
        elif isinstance(layer, LSTM):
            print(f"\nCamada {layer_idx}: LSTM")
            export_lstm_core_12bit(layer, mem, layer_idx, direction=0,
                                   max_neurons=max_neurons, max_idx=max_idx)
            layer_idx += 1
            
        elif isinstance(layer, Dense):
            print(f"\nCamada {layer_idx}: Dense")
            export_dense_core_12bit(layer, mem, layer_idx,
                                    max_neurons=max_neurons, max_idx=max_idx)
            layer_idx += 1
    
    # Salva arquivo
    with open(output_file, "w") as f:
        for addr in sorted(mem.keys()):
            val = np.int32(mem[addr])
            f.write(f"{int(addr):03x} {int(val) & 0xFFFFFFFF:08X}\n")
    
    print(f"\nArquivo salvo: {output_file}")
    print(f"Total de pesos: {len(mem)}")
    print(f"Endereços em 12 bits (formato: 0x000 a 0xFFF)")
    return mem


# ============================================================
# EXPORTA PESOS DA LSTM (core) - 12 BITS
# ============================================================
def export_lstm_core_12bit(lstm_layer, mem, layer_idx, direction=0, 
                           max_neurons=8, max_idx=8):
    """
    Exporta pesos de uma camada LSTM no formato de 12 bits
    """
    weights = lstm_layer.get_weights()
    
    units = lstm_layer.units
    input_dim = weights[0].shape[0]
    
    # Verifica limites
    if units > max_neurons:
        print(f"  ATENÇÃO: units={units} > max_neurons={max_neurons}")
        units = max_neurons
    if input_dim > max_idx:
        print(f"  ATENÇÃO: input_dim={input_dim} > max_idx={max_idx}")
        input_dim = max_idx
    
    kernel = weights[0].T
    recurrent_kernel = weights[1].T
    
    has_bias = len(weights) > 2
    if has_bias:
        bias = weights[2]
    else:
        bias = np.zeros(4 * units)
    
    # Ordem dos gates (mantida a que funciona)
    # gate 00 = input, gate 01 = forget, gate 10 = candidate, gate 11 = output
    gates = ['input', 'forget', 'candidate', 'output']
    
    for gate_idx, gate_name in enumerate(gates):
        # Pesos de entrada (Wx) - recurrent = 0
        kernel_gate = kernel[gate_idx * units:(gate_idx + 1) * units, :]
        for n in range(min(units, max_neurons)):
            for inp in range(min(input_dim, max_idx)):
                addr = build_addr_12bit(
                    layer=layer_idx,
                    is_bias=0,
                    gate=gate_idx,
                    neuron=n,
                    recurrent=0,
                    idx=inp
                )
                mem[addr] = float_to_q(kernel_gate[n, inp])
                
                if n < 2 and inp < 2:
                    print(f"  addr=0x{addr:03X} Wx_{gate_name}[{n}][{inp}] = {kernel_gate[n, inp]:.6f}")
        
        # Pesos recorrentes (Wh) - recurrent = 1
        rec_gate = recurrent_kernel[gate_idx * units:(gate_idx + 1) * units, :]
        for n in range(min(units, max_neurons)):
            for prev_n in range(min(units, max_neurons)):
                addr = build_addr_12bit(
                    layer=layer_idx,
                    is_bias=0,
                    gate=gate_idx,
                    neuron=n,
                    recurrent=1,
                    idx=prev_n
                )
                mem[addr] = float_to_q(rec_gate[n, prev_n])
                
                if n < 2 and prev_n < 2:
                    print(f"  addr=0x{addr:03X} Wh_{gate_name}[{n}][{prev_n}] = {rec_gate[n, prev_n]:.6f}")
        
        # Bias
        bias_gate = bias[gate_idx * units:(gate_idx + 1) * units]
        for n in range(min(units, max_neurons)):
            addr = build_addr_12bit(
                layer=layer_idx,
                is_bias=1,
                gate=gate_idx,
                neuron=n,
                recurrent=0,
                idx=0
            )
            mem[addr] = float_to_q(bias_gate[n])
            
            if n < 2:
                print(f"  addr=0x{addr:03X} bias_{gate_name}[{n}] = {bias_gate[n]:.6f}")


# ============================================================
# EXPORTA PESOS DA DENSE (MLP) - 12 BITS
# ============================================================
def export_dense_core_12bit(dense_layer, mem, layer_idx, max_neurons=8, max_idx=8):
    """Exporta pesos de uma camada Dense no formato de 12 bits"""
    weights = dense_layer.get_weights()
    
    if len(weights) >= 1:
        kernel = weights[0].T
        n_neurons, n_inputs = kernel.shape
        
        if n_neurons > max_neurons:
            print(f"  ATENÇÃO: n_neurons={n_neurons} > max_neurons={max_neurons}")
            n_neurons = max_neurons
        if n_inputs > max_idx:
            print(f"  ATENÇÃO: n_inputs={n_inputs} > max_idx={max_idx}")
            n_inputs = max_idx
        
        # Pesos
        for n in range(n_neurons):
            for inp in range(n_inputs):
                addr = build_addr_12bit(
                    layer=layer_idx,
                    is_bias=0,
                    gate=0,
                    neuron=n,
                    recurrent=0,
                    idx=inp
                )
                mem[addr] = float_to_q(kernel[n, inp])
                
                if n < 3 and inp < 3:
                    print(f"  addr=0x{addr:03X} weight[{n}][{inp}] = {kernel[n, inp]:.6f}")
        
        # Biases
        if len(weights) >= 2:
            bias = weights[1]
            for n in range(n_neurons):
                addr = build_addr_12bit(
                    layer=layer_idx,
                    is_bias=1,
                    gate=0,
                    neuron=n,
                    recurrent=0,
                    idx=0
                )
                mem[addr] = float_to_q(bias[n])
                
                if n < 3:
                    print(f"  addr=0x{addr:03X} bias[{n}] = {bias[n]:.6f}")


# ============================================================
# EXPORTA COMO MLP - 12 BITS
# ============================================================
def export_mlp_weights_12bit(model, output_file="weights.mem", max_neurons=8, max_idx=8):
    """Exporta apenas camadas Dense no formato de 12 bits"""
    print("\n" + "="*50)
    print("MODO MLP: Exportando pesos para 12 bits")
    print("Todas as camadas Dense sao consideradas")
    print("="*50)
    
    mem = {}
    layer_idx = 0
    
    for layer in model.layers:
        if len(layer.get_weights()) == 0:
            continue
            
        if isinstance(layer, Dense):
            print(f"\nCamada {layer_idx}: Dense")
            export_dense_core_12bit(layer, mem, layer_idx, max_neurons, max_idx)
            layer_idx += 1
        elif isinstance(layer, Bidirectional) or isinstance(layer, LSTM):
            print(f"\nAviso: Pulando camada LSTM no modo MLP")
            continue
    
    with open(output_file, "w") as f:
        for addr in sorted(mem.keys()):
            val = np.int32(mem[addr])
            f.write(f"{int(addr):03x} {int(val) & 0xFFFFFFFF:08X}\n")
    
    print(f"\nArquivo salvo: {output_file}")
    print(f"Total de pesos: {len(mem)}")
    return mem


# ============================================================
# FUNCAO PRINCIPAL
# ============================================================
def export_weights(model, mode='lstm', output_file=None, max_neurons=8, max_idx=8):
    """
    Exporta pesos do modelo no formato de 12 bits
    
    Args:
        model: Modelo Keras
        mode: 'lstm' ou 'mlp'
        output_file: Nome do arquivo de saida
        max_neurons: Número máximo de neurônios (padrão 8)
        max_idx: Número máximo de índices (padrão 8)
    """
    if output_file is None:
        output_file = "weights.mem"
    
    if mode == 'lstm':
        return export_lstm_weights(model, output_file, max_neurons, max_idx)
    elif mode == 'mlp':
        return export_mlp_weights_12bit(model, output_file, max_neurons, max_idx)
    else:
        raise ValueError("Modo deve ser 'lstm' ou 'mlp'")


# ============================================================
# FUNÇÕES PARA EXPORTAR DADOS DE TESTE (Q8.24)
# ============================================================
def export_matrix_to_mem_12bit(matrix, result=None, ground_result=None, output_file="matrix.mem"):
    """Exporta matriz para formato Q8.24 com endereços de 12 bits"""
    mem = {}
    
    for linha_idx in range(matrix.shape[0]):
        for col_idx in range(4):
            addr = (linha_idx << 2) | col_idx
            # Garante que o endereço cabe em 12 bits
            if addr > 0xFFF:
                print(f"ATENÇÃO: addr=0x{addr:04X} excede 12 bits!")
            valor = matrix[linha_idx, col_idx]
            valor_q = float_to_q(valor)
            mem[addr] = valor_q
    
    with open(output_file, "w") as f:
        line = 0
        result_hex = float_to_q(result) if result is not None else 0
        gt_hex = ground_result if ground_result is not None else 0
        
        for addr in sorted(mem.keys()):
            val = np.int32(mem[addr])
            if line == 0 and result is not None:
                f.write(f"{int(addr):03x} {int(val) & 0xFFFFFFFF:08X} {int(result_hex) & 0xFFFFFFFF:08X}  {int(gt_hex) & 0xFFFFFFFF:01X}   \n")
            else:
                f.write(f"{int(addr):03x} {int(val) & 0xFFFFFFFF:08X}\n")
            line += 1
    
    print(f"Arquivo salvo: {output_file}")
    return mem


def float_to_q8_24(value):
    """Converte float para Q8.24 (formato signed 32 bits)"""
    scaled = value * SCALE
    rounded = int(round(scaled))

    if rounded > 2**31 - 1:
        rounded = 2**31 - 1
    elif rounded < -2**31:
        rounded = -2**31

    return rounded


def gerar_x_array_2d(valores):
    """
    Gera código SystemVerilog para preencher array x com valores Q8.24
    
    Exemplo de saída:
        x[0][0] =  32'sd16777216;   // 1.0
        x[0][1] = -32'sd8388608;    // -0.5
    """
    if len(valores) == 1 and isinstance(valores[0][0], list):
        valores = valores[0]

    linhas = []

    for i, linha in enumerate(valores):
        linha_atual = ""
        for j, valor in enumerate(linha):
            qval = float_to_q8_24(valor)
            
            if qval >= 0:
                atrib = f"x[{i}][{j}] =  32'sd{qval};"
            else:
                atrib = f"x[{i}][{j}] = -32'sd{abs(qval)};"
            
            atrib += f"  // {valor:.6f}"
            linha_atual += f"{atrib}   "

        linhas.append(linha_atual.rstrip())

    return "\n".join(linhas)
