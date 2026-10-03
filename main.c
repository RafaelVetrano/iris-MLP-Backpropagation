/*
 * MLP 4-3-3 treinada com backpropagation para classificar a base Iris.
 *
 *
 *
 * Notacao:
 *   I = net do neuronio (soma ponderada), O = saida apos a ativacao,
 *   d = delta (erro local), t = saida desejada (one-hot).
 *   Os indices comecam em 1; o bias ocupa a ultima posicao de cada camada
 *   (I0[5] na entrada e I1[4] na oculta).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
    int i = 0;
    int j, l;    /* i: entrada, j: oculta, l: saida */
    int n, m;    /* m: epoca, n: iteracao */
    int cs;      /* amostra sorteada */
    float er;    /* erro quadratico medio da epoca */

    float I0[15], O0[15], I1[15], O1[15], I2[13], O2[13];
    float w1[15][14], w2[15][13];     /* pesos atuais */
    float nw1[15][14], nw2[15][13];   /* pesos novos */
    float vw1[15][14], vw2[15][13];   /* variacao anterior (momentum) */
    float d2[13], d1[14];             /* deltas da saida e da oculta */
    float loss;
    int acertos, prev, real;

    float E1[200], E2[200], E3[200], E4[200];   /* entradas */
    float t1[200], t2[200], t3[200];            /* saidas desejadas */

    /* Leitura dos dados normalizados: E1 E2 E3 E4 t1 t2 t3 por linha */
    FILE *in;

    if ((in = fopen("data_iris_normalizado.txt", "rt")) == NULL) {
        printf("Cannot open input file.\n");
        return 1;
    }

    while (!feof(in)) {
        fscanf(in, "%f%f%f%f%f%f%f", &E1[i], &E2[i], &E3[i], &E4[i], &t1[i], &t2[i], &t3[i]);
        i++;
    }

    /* i-1: o laco com feof conta uma leitura a mais */
    for (j = 0; j < (i - 1); j++) {
        printf("%d %f %f %f %f %f %f %f\n", j, E1[j], E2[j], E3[j], E4[j], t1[j], t2[j], t3[j]);
    }

    fclose(in);

    /* Inicializacao dos pesos em [-5.0, 4.9] */
    for (j = 1; j <= 3; j++) {
        for (i = 1; i <= 5; i++) {
            w1[i][j] = ((rand() % 100) / 10.0) - 5.0;
            vw1[i][j] = 0.0;
        }
    }

    for (l = 1; l <= 3; l++) {
        for (j = 1; j <= 4; j++) {
            w2[j][l] = ((rand() % 100) / 10.0) - 5.0;
            vw2[j][l] = 0.0;
        }
    }

    /* Treinamento online: pesos ajustados apos cada amostra sorteada */
    for (m = 0; m <= 1000; m++) {
        er = 0.0;
        for (n = 0; n <= 1500; n++) {
            /* Forward */
            cs = rand() % 150;

            I0[1] = E1[cs];
            I0[2] = E2[cs];
            I0[3] = E3[cs];
            I0[4] = E4[cs];
            I0[5] = 1.0;   /* bias */

            O0[1] = I0[1];
            O0[2] = I0[2];
            O0[3] = I0[3];
            O0[4] = I0[4];
            O0[5] = I0[5];

            for (j = 1; j <= 3; j++) {
                I1[j] = 0.0;
                for (i = 1; i <= 5; i++) {
                    I1[j] += O0[i] * w1[i][j];
                }
                O1[j] = 1.0 / (1.0 + exp(-I1[j]));
            }
            I1[4] = 1.0;   /* bias da oculta */
            O1[4] = I1[4];

            for (l = 1; l <= 3; l++) {
                I2[l] = 0.0;
                for (j = 1; j <= 4; j++) {
                    I2[l] += O1[j] * w2[j][l];
                }
                O2[l] = 1.0 / (1.0 + exp(-I2[l]));
            }

            /* Backward: d2 = (t - O) * O * (1 - O) */
            d2[1] = (t1[cs] - O2[1]) * O2[1] * (1.0 - O2[1]);
            d2[2] = (t2[cs] - O2[2]) * O2[2] * (1.0 - O2[2]);
            d2[3] = (t3[cs] - O2[3]) * O2[3] * (1.0 - O2[3]);

            /* d1 = O * (1 - O) * soma(d2 * w2), usando w2 antes da atualizacao */
            d1[1] = O1[1] * (1.0 - O1[1]) * (d2[1] * w2[1][1] + d2[2] * w2[1][2] + d2[3] * w2[1][3]);
            d1[2] = O1[2] * (1.0 - O1[2]) * (d2[1] * w2[2][1] + d2[2] * w2[2][2] + d2[3] * w2[2][3]);
            d1[3] = O1[3] * (1.0 - O1[3]) * (d2[1] * w2[3][1] + d2[2] * w2[3][2] + d2[3] * w2[3][3]);

            /* Atualizacao: w += eta * d * O + alfa * dw_anterior (eta = alfa = 0.5) */
            for (l = 1; l <= 3; l++) {
                for (j = 1; j <= 4; j++) {
                    nw2[j][l] = w2[j][l] + 0.5 * d2[l] * O1[j] + 0.5 * vw2[j][l];
                    vw2[j][l] = nw2[j][l] - w2[j][l];
                    w2[j][l] = nw2[j][l];
                }
            }

            for (j = 1; j <= 3; j++) {
                for (i = 1; i <= 5; i++) {
                    nw1[i][j] = w1[i][j] + 0.5 * d1[j] * O0[i] + 0.5 * vw1[i][j];
                    vw1[i][j] = nw1[i][j] - w1[i][j];
                    w1[i][j] = nw1[i][j];
                }
            }

            er += (t1[cs] - O2[1]) * (t1[cs] - O2[1])
                + (t2[cs] - O2[2]) * (t2[cs] - O2[2])
                + (t3[cs] - O2[3]) * (t3[cs] - O2[3]);
        }
        er = er / 1500.0;
        printf("%d %f \n", m, er);
    }

    /* Pesos finais (uma linha por neuronio de destino) */
    for (j = 1; j <= 3; j++) {
        for (i = 1; i <= 5; i++) {
            printf("%f ", w1[i][j]);
        }
        printf("\n");
    }

    for (l = 1; l <= 3; l++) {
        for (j = 1; j <= 4; j++) {
            printf("%f ", w2[j][l]);
        }
        printf("\n");
    }

    /* Avaliacao: apenas forward em todas as amostras */
    loss = 0.0;
    acertos = 0;
    for (m = 0; m <= 149; m++) {
        I0[1] = E1[m];
        I0[2] = E2[m];
        I0[3] = E3[m];
        I0[4] = E4[m];
        I0[5] = 1.0;

        O0[1] = I0[1];
        O0[2] = I0[2];
        O0[3] = I0[3];
        O0[4] = I0[4];
        O0[5] = I0[5];

        for (j = 1; j <= 3; j++) {
            I1[j] = 0.0;
            for (i = 1; i <= 5; i++) {
                I1[j] += O0[i] * w1[i][j];
            }
            O1[j] = 1.0 / (1.0 + exp(-I1[j]));
        }
        I1[4] = 1.0;
        O1[4] = I1[4];

        for (l = 1; l <= 3; l++) {
            I2[l] = 0.0;
            for (j = 1; j <= 4; j++) {
                I2[l] += O1[j] * w2[j][l];
            }
            O2[l] = 1.0 / (1.0 + exp(-I2[l]));
        }

        printf("%d %f %f %f %f %f %f %f \n", m, E1[m], E2[m], E3[m], E4[m], O2[1], O2[2], O2[3]);

        loss += (t1[m] - O2[1]) * (t1[m] - O2[1])
              + (t2[m] - O2[2]) * (t2[m] - O2[2])
              + (t3[m] - O2[3]) * (t3[m] - O2[3]);

        /* Classe prevista e real por argmax */
        prev = 1;
        if (O2[2] > O2[prev]) prev = 2;
        if (O2[3] > O2[prev]) prev = 3;

        real = 1;
        if (t2[m] > t1[m] && t2[m] > t3[m]) real = 2;
        if (t3[m] > t1[m] && t3[m] > t2[m]) real = 3;

        if (prev == real) acertos++;
    }

    loss = loss / 150.0;

    printf("\n==================== RESULTADO FINAL ====================\n");
    printf("Loss (treino, ultima epoca): %f\n", er);
    printf("Loss (avaliacao final)     : %f\n", loss);
    printf("Acuracia                   : %d/150 = %.2f%%\n", acertos, 100.0 * acertos / 150.0);
    printf("=========================================================\n");

    return 0;
}
