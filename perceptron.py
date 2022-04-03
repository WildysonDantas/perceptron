import pandas as pd
import numpy as np
import random as rd
import copy


class Perceptron():

    def __init__(self, n, b):
        self.n = n
        self.b = b
        self.w = [rd.uniform(0, 1), rd.uniform(0, 1), rd.uniform(0, 1)]

    # sinal de ativação
    def sinal(self, u):
        if(u >= 0):
            return 1
        else:
            return -1

    # classe treinamento
    def trainning(self, x, d):

        print("Treinamento")
        print(".\n.\n.\n")
        w2 = []
        w2.append(copy.deepcopy(self.w))
        # contador de epocas
        num_epocas = 1
        # para todas as amostras de treinamento
        for i in range(0, len(x)):

            erro = False

            while True:
                u = 0
                for j in range(0, len(self.w)):
                    u += self.w[j] * x[i][j]

                u = u + self.b
                # função de ativação
                y = self.sinal(u)

                # verifica se as saidas
                if y != d[i]:

                    # taxa de erros entre a saida de desejada e a saida calculada
                    taxa_error = d[i] - y

                    # faz o ajuste dos pesos
                    for j in range(0, len(self.w)):
                        self.w[j] = self.w[j] + self.n * (taxa_error * x[i][j])
                    # print("ocorreu um erro. na amostra:",
                        #  x[i], " Atulizando pesos", ' Pesos ', self.w)
                        print("atualizando pesos da amostra", i+1)

                    erro = True

                elif y == -1:
                    # print("Classe P1. Amostra: ",
                      #    x[i], 'Saida Desejada ',  d[i], "Saida Gerada: ", y, ' Peso ', self.w)
                    erro = False
                elif y == 1:
                    # print("Classe P2. Amostra: ",
                     #     x[i], 'Saida Desejada ',  d[i], "Saida Gerada: ", y, ' Peso', self.w)

                    erro = False

                # pegando todos os aclores de pesos ja utilizados
                w2.append(copy.deepcopy(self.w))
                num_epocas += 1

                # caso nao tenha erro parar
                if not erro:
                    print("acerto")
                    break

        print("Numero de epocas", num_epocas)
       # print("Treinamento Finalizado\nPpesos utilizados")
        c = 1
        # for ele in w2:

        #  print(c, ele)
        #   c += 1

    def predict(self, data, classe1, classe2):

        # utiliza o vetor de pesos que foi ajustado na fase de treinamento
        for i in range(len(data)):
            u = 0
            for j in range(len(self.w)):
                u += self.w[j] * data[i][j]
            u = u + self.b
            # calcula a saída
            y = self.sinal(u)
            # bias

            # verifica a qual classe pertence
            if y == -1:
                print(i, 'A amostra {pertence a classe %s' %
                      classe1, 'saida', y)
            else:
                print(i, 'A amostra  pertence a classe %s' %
                      classe2, 'saida', y)

    def main(self):

        df = pd.read_excel('treinamento.xls')
        d = df.head(30).to_numpy()[:, 3]
        x = df.head(30).to_numpy()
        x = np.delete(x, 3, 1)
        # print(df)
       # print(d)
        #print("\n", x)

        self.trainning(x, d)

        dfdata = pd.read_excel('data.xlsx')
        data = dfdata.to_numpy()
        self.predict(data, "P1", "P2")
        # print(data)


p = Perceptron(0.01, 1)
p.main()
