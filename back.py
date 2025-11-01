import numpy as np

# =============================================================
# Funções auxiliares
# =============================================================
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# =============================================================
# Rede Neural - Problema do XOR
# =============================================================
class NeuralNetworkXOR:
    def __init__(self, taxa_aprendizado=0.5):
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 1
        self.lr = taxa_aprendizado

        # Pesos inicializados aleatoriamente
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        output = sigmoid(self.z2)
        return output

    def backpropagation(self, X, y, output):
        erro_saida = y - output
        delta_saida = erro_saida * sigmoid(output, deriv=True)

        erro_oculta = delta_saida.dot(self.W2.T)
        delta_oculta = erro_oculta * sigmoid(self.a1, deriv=True)

        # Equação de ajuste dos pesos:
        # ΔW = η * (entrada^T · δ)
        self.W2 += self.lr * self.a1.T.dot(delta_saida)
        self.W1 += self.lr * X.T.dot(delta_oculta)

    def treinar(self, X, y, epocas=10000):
        for i in range(epocas):
            output = self.feedforward(X)
            self.backpropagation(X, y, output)

            if i % 1000 == 0:
                mse = np.mean((y - output) ** 2)
                print(f"Época {i} - Erro MSE: {mse:.6f}")

    def testar(self, X, y):
        print("\n--- Resultados Finais (XOR) ---")
        output = self.feedforward(X)
        for entrada, esperado, obtido in zip(X, y, output):
            print(f"Entrada: {entrada} | Esperado: {esperado} | Obtido: {obtido.round(3)}")


# =============================================================
# Rede Neural - Display de 7 Segmentos (7-5-4)
# =============================================================
class NeuralNetwork7Seg:
    def __init__(self, taxa_aprendizado=0.3):
        self.input_size = 7
        self.hidden_size = 5
        self.output_size = 4
        self.lr = taxa_aprendizado

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        saida = sigmoid(self.z2)
        return saida

    def backpropagation(self, X, y, output):
        erro_saida = y - output
        delta_saida = erro_saida * sigmoid(output, deriv=True)

        erro_oculta = delta_saida.dot(self.W2.T)
        delta_oculta = erro_oculta * sigmoid(self.a1, deriv=True)

        # Equação de ajuste dos pesos:
        # ΔW = η * (entrada^T · δ)
        self.W2 += self.lr * self.a1.T.dot(delta_saida)
        self.W1 += self.lr * X.T.dot(delta_oculta)

    def treinar(self, X, y, epocas=5000):
        for i in range(epocas):
            output = self.feedforward(X)
            self.backpropagation(X, y, output)

            if i % 500 == 0:
                mse = np.mean((y - output) ** 2)
                print(f"Época {i} - Erro MSE: {mse:.6f}")

    def testar(self, entradas, saidas):
        print("\n--- Resultados Finais (Display 7 Segmentos) ---")
        for i in range(len(entradas)):
            saida = self.feedforward(entradas[i])
            print(f"Dígito: {i} | Saída obtida: {np.round(saida)} | Esperado: {saidas[i]}")

        # Teste com ruído (um segmento desligado)
        print("\n--- Teste com ruído (1 segmento desligado aleatoriamente) ---")
        for i in range(10):
            teste = entradas[i].copy()
            idx_ruido = np.random.randint(0, 7)
            teste[idx_ruido] = 0
            saida = self.feedforward(teste)
            print(f"Dígito real: {i}, Segmento com falha: {idx_ruido}, Saída obtida: {np.round(saida)}")


# =============================================================
# Menu principal (tipo switch-case)
# =============================================================
def main():
    print("====================================")
    print("       LISTA #8 - Redes Neurais     ")
    print("====================================")
    print("Escolha o problema:")
    print("1 - Problema do XOR")
    print("2 - Dígitos do Display de 7 Segmentos")
    print("====================================")

    opcao = input("Digite a opção desejada (1 ou 2): ")

    if opcao == "1":
        # ----- Problema do XOR -----
        X = np.array([[0,0],
                      [0,1],
                      [1,0],
                      [1,1]])
        y = np.array([[0],
                      [1],
                      [1],
                      [0]])

        rede = NeuralNetworkXOR()
        rede.treinar(X, y)
        rede.testar(X, y)

    elif opcao == "2":
        # ----- Display de 7 Segmentos -----
        entradas = np.array([
            [1,1,1,1,1,1,0],  # 0
            [0,1,1,0,0,0,0],  # 1
            [1,1,0,1,1,0,1],  # 2
            [1,1,1,1,0,0,1],  # 3
            [0,1,1,0,0,1,1],  # 4
            [1,0,1,1,0,1,1],  # 5
            [1,0,1,1,1,1,1],  # 6
            [1,1,1,0,0,0,0],  # 7
            [1,1,1,1,1,1,1],  # 8
            [1,1,1,1,0,1,1]   # 9
        ], dtype=float)

        saidas = np.array([
            [0,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
            [0,0,1,1],
            [0,1,0,0],
            [0,1,0,1],
            [0,1,1,0],
            [0,1,1,1],
            [1,0,0,0],
            [1,0,0,1]
        ], dtype=float)

        rede = NeuralNetwork7Seg()
        rede.treinar(entradas, saidas)
        rede.testar(entradas, saidas)

    else:
        print("Opção inválida! Encerrando programa.")


# =============================================================
# Execução principal
# =============================================================
if __name__ == "__main__":
    main()
