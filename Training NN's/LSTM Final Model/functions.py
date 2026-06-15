
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


FRAC = 16
SCALE = 1 << FRAC

INT32_MAX = np.int32(2**31 - 1)
INT32_MIN = np.int32(-(2**31))

def float_to_q16(val):
    if np.isnan(val):
        return np.int32(0)
    if np.isposinf(val):
        return INT32_MAX
    if np.isneginf(val):
        return INT32_MIN
    fixed = np.round(val * SCALE)
    return np.int32(fixed)

# ============================================================
# ENDERECO DE 21 BITS
# ============================================================
def build_addr(layer, is_bias, is_lstm, gate, neuron, recurrent, idx):
    addr = 0
    addr |= (layer     & 0xF)   << 17  # bits 20-17: camada (max 16)
    addr |= (is_bias   & 0x1)   << 16  # bit 16:    bias
    addr |= (is_lstm   & 0x1)   << 15  # bit 15:    lstm
    addr |= (gate      & 0x3)   << 13  # bits 14-13: porta LSTM
    addr |= (neuron    & 0x3F)  << 7   # bits 12-7:  neuronio (max 64)
    addr |= (recurrent & 0x1)   << 6   # bit 6:     recorrente
    addr |= (idx       & 0x3F)        # bits 5-0:   indice (max 64)
    return addr

# ============================================================
# EXPORTA PESOS DA LSTM (ignorando Embedding)
# ============================================================
def export_lstm_weights(model, output_file="weights_lstm.mem"):
    print("\n" + "="*50)
    print("MODO LSTM: Exportando pesos a partir da primeira LSTM")
    print("Ignorando: Embedding e SpatialDropout1D")
    print("="*50)
    
    mem = {}
    layer_idx = 0  # camada 0 = primeira LSTM
    
    # Percorre as camadas do modelo
    for layer in model.layers:
        # Pula camadas que nao tem pesos
        if len(layer.get_weights()) == 0:
            continue
            
        # Verifica se é LSTM (incluindo Bidirectional)
        if isinstance(layer, Bidirectional):
            # Bidirectional tem 2 LSTMs (forward e backward)
            print(f"\nCamada {layer_idx}: Bidirectional LSTM")
            
            forward_lstm = layer.forward_layer
            backward_lstm = layer.backward_layer
            
            # Exporta forward LSTM
            export_lstm_core(forward_lstm, mem, layer_idx, direction=0)
            # Exporta backward LSTM (proxima camada)
            export_lstm_core(backward_lstm, mem, layer_idx + 1, direction=1)
            
            layer_idx += 2  # Duas camadas (forward e backward)
            
        elif isinstance(layer, LSTM):
            print(f"\nCamada {layer_idx}: LSTM")
            export_lstm_core(layer, mem, layer_idx, direction=0)
            layer_idx += 1
            
        elif isinstance(layer, Dense):
            print(f"\nCamada {layer_idx}: Dense")
            export_dense_core(layer, mem, layer_idx)
            layer_idx += 1
    
    # Salva arquivo
    with open(output_file, "w") as f:
        for addr in sorted(mem.keys()):
            val = np.int32(mem[addr])
            f.write(f"{int(addr):05x} {int(val) & 0xFFFFFFFF:08X}\n")
    
    print(f"\nArquivo salvo: {output_file}")
    print(f"Total de pesos: {len(mem)}")
    return mem

# ============================================================
# EXPORTA PESOS DA LSTM (core)
# ============================================================
def export_lstm_core(lstm_layer, mem, layer_idx, direction=0):
    """
    Exporta pesos de uma camada LSTM
    direction: 0 = forward, 1 = backward (apenas para identificacao)
    """
    weights = lstm_layer.get_weights()
    
    # LSTM tem 3 matrizes de pesos e 3 biases (ou 2 se usar bias=False)
    # weights[0]: kernel (input weights) - formato: (input_dim, 4 * units)
    # weights[1]: recurrent_kernel (recurrent weights) - formato: (units, 4 * units)
    # weights[2]: bias (se existir) - formato: (4 * units,)
    
    units = lstm_layer.units
    input_dim = weights[0].shape[0]
    
    # Separa os pesos para cada porta (forget, input, candidate, output)
    # Ordem: i, f, c, o (input, forget, candidate, output)
    kernel = weights[0].T  # Transpoe para (4*units, input_dim)
    recurrent_kernel = weights[1].T  # Transpoe para (4*units, units)
    
    # Bias
    has_bias = len(weights) > 2
    if has_bias:
        bias = weights[2]
    else:
        bias = np.zeros(4 * units)
    
    # Para cada porta
    #gates = ['forget', 'input', 'candidate', 'output']
    gates = ['candidate', 'forget', 'input', 'output']
    
    for gate_idx, gate_name in enumerate(gates):
        # Pesos de entrada (Wx)
        kernel_gate = kernel[gate_idx * units:(gate_idx + 1) * units, :]
        for n in range(units):
            for inp in range(input_dim):
                addr = build_addr(
                    layer=layer_idx,
                    is_bias=0,
                    is_lstm=1,
                    gate=gate_idx,
                    neuron=n,
                    recurrent=0,
                    idx=inp
                )
                mem[addr] = float_to_q16(kernel_gate[n, inp])
                
                # Mostra alguns exemplos (opcional)
                if n < 2 and inp < 2:
                    print(f"  addr=0x{addr:05X} Wx_{gate_name}[{n}][{inp}] = {kernel_gate[n, inp]:.6f}")
        
        # Pesos recorrentes (Wh)
        rec_gate = recurrent_kernel[gate_idx * units:(gate_idx + 1) * units, :]
        for n in range(units):
            for prev_n in range(units):
                addr = build_addr(
                    layer=layer_idx,
                    is_bias=0,
                    is_lstm=1,
                    gate=gate_idx,
                    neuron=n,
                    recurrent=1,
                    idx=prev_n
                )
                mem[addr] = float_to_q16(rec_gate[n, prev_n])
                
                if n < 2 and prev_n < 2:
                    print(f"  addr=0x{addr:05X} Wh_{gate_name}[{n}][{prev_n}] = {rec_gate[n, prev_n]:.6f}")
        
        # Bias
        bias_gate = bias[gate_idx * units:(gate_idx + 1) * units]
        for n in range(units):
            addr = build_addr(
                layer=layer_idx,
                is_bias=1,
                is_lstm=1,
                gate=gate_idx,
                neuron=n,
                recurrent=0,
                idx=0
            )
            mem[addr] = float_to_q16(bias_gate[n])
            
            if n < 2:
                print(f"  addr=0x{addr:05X} bias_{gate_name}[{n}] = {bias_gate[n]:.6f}")

# ============================================================
# EXPORTA PESOS DA DENSE (MLP comum)
# ============================================================
def export_dense_core(dense_layer, mem, layer_idx):
    """Exporta pesos de uma camada Dense"""
    weights = dense_layer.get_weights()
    
    if len(weights) >= 1:
        kernel = weights[0].T  # Transpoe para (output_dim, input_dim)
        n_neurons, n_inputs = kernel.shape
        
        # Pesos
        for n in range(n_neurons):
            for inp in range(n_inputs):
                addr = build_addr(
                    layer=layer_idx,
                    is_bias=0,
                    is_lstm=0,  # Nao é LSTM
                    gate=0,
                    neuron=n,
                    recurrent=0,
                    idx=inp
                )
                mem[addr] = float_to_q16(kernel[n, inp])
                
                if n < 3 and inp < 3:
                    print(f"  addr=0x{addr:05X} weight[{n}][{inp}] = {kernel[n, inp]:.6f}")
        
        # Biases
        if len(weights) >= 2:
            bias = weights[1]
            for n in range(n_neurons):
                addr = build_addr(
                    layer=layer_idx,
                    is_bias=1,
                    is_lstm=0,
                    gate=0,
                    neuron=n,
                    recurrent=0,
                    idx=0
                )
                mem[addr] = float_to_q16(bias[n])
                
                if n < 3:
                    print(f"  addr=0x{addr:05X} bias[{n}] = {bias[n]:.6f}")

# ============================================================
# EXPORTA COMO MLP (ignorando LSTM)
# ============================================================
def export_mlp_weights(model, output_file="weights_mlp.mem"):
    print("\n" + "="*50)
    print("MODO MLP: Exportando pesos como rede neural comum")
    print("Todas as camadas Dense sao consideradas")
    print("="*50)
    
    mem = {}
    layer_idx = 0
    
    for layer in model.layers:
        if len(layer.get_weights()) == 0:
            continue
            
        if isinstance(layer, Dense):
            print(f"\nCamada {layer_idx}: Dense")
            export_dense_core(layer, mem, layer_idx)
            layer_idx += 1
        elif isinstance(layer, Bidirectional) or isinstance(layer, LSTM):
            print(f"\nAviso: Pulando camada LSTM no modo MLP")
            continue
    
    with open(output_file, "w") as f:
        for addr in sorted(mem.keys()):
            val = np.int32(mem[addr])
            f.write(f"{int(addr):05x} {int(val) & 0xFFFFFFFF:08X}\n")
    
    print(f"\nArquivo salvo: {output_file}")
    print(f"Total de pesos: {len(mem)}")
    return mem

# ============================================================
# FUNCAO PRINCIPAL
# ============================================================
def export_weights(model, mode='lstm', output_file=None):
    """
    Exporta pesos do modelo
    
    Args:
        model: Modelo Keras
        mode: 'lstm' (ignora Embedding) ou 'mlp' (apenas camadas Dense)
        output_file: Nome do arquivo de saida
    """
    if mode == 'lstm':
        if output_file is None:
            output_file = "weights_lstm.mem"
        return export_lstm_weights(model, output_file)
    elif mode == 'mlp':
        if output_file is None:
            output_file = "weights_mlp.mem"
        return export_mlp_weights(model, output_file)
    else:
        raise ValueError("Modo deve ser 'lstm' ou 'mlp'")

def export_matrix_to_mem(matrix, output_file="matrix.mem"):
    mem = {}
    
    for linha_idx in range(matrix.shape[0]):
        for col_idx in range(4):
            addr = (linha_idx << 2) | col_idx
            valor = matrix[linha_idx, col_idx]
            valor_q16 = float_to_q16(valor)  # sua função original
            mem[addr] = valor_q16
    
    # ESCREVE EXATAMENTE COMO NO SEU CÓDIGO ORIGINAL
    with open(output_file, "w") as f:
        for addr in sorted(mem.keys()):
            val = np.int32(mem[addr])  # ← igual ao original
            f.write(f"{int(addr):05x} {int(val) & 0xFFFFFFFF:08X}\n")
    
    print(f"Arquivo salvo: {output_file}")
    return mem


def float_to_q16_16(value):
    """Converte float para Q16.16 (formato signed 32 bits)"""
    scaled = value * 65536
    rounded = int(round(scaled))

    if rounded > 2**31 - 1:
        rounded = 2**31 - 1
    elif rounded < -2**31:
        rounded = -2**31

    return rounded


def gerar_x_array_2d(valores):
    """
    Gera código:
    x[i][j] = ±32'sdN;

    Aceita:
        shape (N,M)
        ou (1,N,M)
    """

    # remove dimensão batch se existir
    if len(valores) == 1 and isinstance(valores[0][0], list):
        valores = valores[0]

    linhas = []

    for i, linha in enumerate(valores):

        linha_atual = ""

        for j, valor in enumerate(linha):

            qval = float_to_q16_16(valor)

            if qval >= 0:
                atrib = f"x[{i}][{j}] =  32'sd{qval};"
            else:
                atrib = f"x[{i}][{j}] = -32'sd{abs(qval)};"

            linha_atual += f"{atrib}   "

        linhas.append(linha_atual.rstrip())

    return "\n".join(linhas)