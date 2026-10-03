# Iris MLP Backpropagation

Rede neural MLP (Multi-Layer Perceptron) implementada do zero em C, treinada com backpropagation para classificar as 3 espécies da base [Iris](https://archive.ics.uci.edu/dataset/53/iris).

Sem bibliotecas externas: só `stdio.h`, `stdlib.h` e `math.h`.

Projeto acadêmico desenvolvido na disciplina de **Sistemas Inteligentes** da **UNAERP** (Universidade de Ribeirão Preto).

## Arquitetura

```
Entrada (4 + bias)  ->  Oculta (3 + bias)  ->  Saída (3)
                         sigmoide               sigmoide
```

| Item | Valor |
|---|---|
| Entradas | 4 medidas da flor (sépala e pétala), normalizadas em [0, 1] |
| Saídas | 3 neurônios, codificação one-hot (`1 0 0`, `0 1 0`, `0 0 1`) |
| Ativação | Sigmoide |
| Treinamento | Online (estocástico), com amostras sorteadas a cada iteração |
| Taxa de aprendizado (η) | 0.5 |
| Momentum (α) | 0.5 |
| Épocas | 1001 × 1501 iterações |
| Função de erro | Erro quadrático médio |

## Como funciona

A rede recebe as 4 medidas de uma flor e responde com 3 números, um para cada espécie. A espécie com o maior número é o palpite da rede.

No começo os pesos são aleatórios, então a rede erra muito. O treino repete três passos milhares de vezes:

1. **Palpite:** uma flor é sorteada da base e a rede calcula sua resposta.
2. **Erro:** a resposta é comparada com a espécie correta. O erro calculado na saída é enviado de volta para a camada oculta. Daí vem o nome *backpropagation*.
3. **Ajuste:** cada peso é corrigido um pouco na direção que diminui o erro. O momentum aproveita parte do ajuste anterior para a rede aprender mais rápido.

Com o tempo o erro cai e a rede aprende a separar as espécies. No final, ela é testada com as 150 flores e o programa mostra quantas acertou.

## Como executar

```bash
gcc main.c -o main -lm
./main
```

O arquivo `data_iris_normalizado.txt` precisa estar no diretório em que o programa é executado. Cada linha tem o formato:

```
E1 E2 E3 E4 t1 t2 t3
```

## Saída

O programa imprime:

- os dados lidos;
- o erro médio de cada época (deve diminuir ao longo do treino);
- os pesos finais das duas camadas;
- a resposta da rede para cada amostra;
- um resumo com a loss e a acurácia.

Exemplo de resultado:

```
==================== RESULTADO FINAL ====================
Loss (treino, ultima epoca): 0.015825
Loss (avaliacao final)     : 0.018032
Acuracia                   : 149/150 = 99.33%
=========================================================
```

> A avaliação usa os mesmos dados do treino, então a acurácia mede o ajuste, não a generalização. Como `rand()` não recebe seed, o resultado é reprodutível no mesmo compilador e sistema, mas pode variar em outros.

## Créditos

Baseado nas aulas de Sistemas inteligentes por Milton Faria Junior (UNAERP).
Base de dados: Fisher, R. A. (1936), [*Iris*](https://archive.ics.uci.edu/dataset/53/iris), UCI Machine Learning Repository .
