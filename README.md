# Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)

Trabalho da disciplina de **Sistemas Inteligentes e Aprendizado de Máquina** (PUC-Campinas).
Implementação em Python de um algoritmo genético (AG) que resolve, de forma local e heurística, o problema do caixeiro viajante em 2D, seguindo as estratégias vistas em aula.

## O problema

- Pontos em duas dimensões.
- Encontrar um **caminho cíclico** que passe por todos os pontos, minimizando a **distância euclidiana** total.
- Quantidade de pontos arbitrária (mínimo 8).
- Dois cenários:
  1. **Aleatório:** pontos distribuídos uniformemente em [0, 100]². O ótimo é desconhecido.
  2. **Círculo (benchmark):** pontos igualmente espaçados num círculo. O ótimo é conhecido (perímetro do polígono regular), então dá para medir o erro.
- Bônus: teste com muitos pontos em círculo, anotando tempo de convergência e soluções em épocas intermediárias.

## Requisitos e instalação

Python 3.8 ou superior.

```bash
python -m pip install numpy matplotlib
```

## Como executar

```bash
python tsp_ga.py                    # cenários aleatório e círculo (30 pontos)
python tsp_ga.py --n 40 --seed 7    # muda o número de pontos e a semente
python tsp_ga.py --mut-rate 0.1     # muda a taxa de mutação
python tsp_ga.py --explorar         # compara várias taxas de mutação
python tsp_ga.py --bonus 100        # teste bônus: 100 pontos em círculo
python tsp_ga.py --gif              # salva GIFs da evolução
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--n` | 30 | Pontos dos cenários base (mínimo 8) |
| `--seed` | 42 | Semente aleatória (reprodutibilidade) |
| `--mut-rate` | 0.05 | Taxa de mutação |
| `--bonus N` | 0 (desligado) | Roda também um círculo com N pontos |
| `--explorar` | desligado | Compara taxas de mutação no cenário aleatório |
| `--gif` | desligado | Gera GIFs da evolução |

### Saídas (pasta `resultados/`)

| Arquivo | Conteúdo |
|---|---|
| `aleatorio.png`, `circulo.png`, `bonus_circulo.png` | Rotas em épocas selecionadas e curva de convergência |
| `aleatorio.csv`, `circulo.csv`, `bonus_circulo.csv` | Melhor, média e pior distância e tempo de **todas as épocas** |
| `exploracao_mutacao.png` | Curvas de convergência para cada taxa de mutação (`--explorar`) |
| `*.gif` | Animação da evolução (`--gif`) |

O terminal também imprime o progresso periodicamente, uma tabela de desempenho por época e, no círculo, o gap em relação ao ótimo.

## Como o algoritmo funciona

Cada ciclo é chamado de **época**:

1. Calcula a distância (aptidão) de cada indivíduo e faz o ranking.
2. Verifica os critérios de parada.
3. Seleciona os pais.
4. Gera a nova população por cruzamento e mutação.
5. Repete.

### Escolhas de projeto

| Componente | Escolha | Justificativa |
|---|---|---|
| **Gene** | Permutação dos índices das cidades, ex.: `[A,E,C,D,B,F]` | Toda cidade aparece exatamente uma vez, então nunca existe solução inválida. A rota é cíclica (a última cidade volta à primeira). |
| **Aptidão** | `F(I)` = soma das distâncias entre cidades adjacentes + distância da última à primeira. O AG **minimiza** F. | É a função definida na aula. A restrição de visitar todas as cidades fica na própria genética. |
| **População** | Tamanho fixo de `6 × n`, limitado entre 100 e 600 | Mais cidades exigem mais variedade genética (espaço de busca de (n−1)!/2 rotas). População grande evita mínimos locais, mas custa mais por época. |
| **Seleção** | Torneio com k=5, mais elitismo dos 2 melhores | O torneio controla a pressão seletiva e não depende da escala da aptidão. O elitismo mantém as soluções boas até serem superadas. |
| **Cruzamento** | Ordered Crossover: mantém a primeira metade de um pai e completa com as cidades faltantes na ordem do outro pai. Probabilidade de 90% por par. | Garante permutações válidas. Crossovers de 1 ponto ou uniforme gerariam cidades repetidas ou faltando. |
| **Mutação** | Inversão de segmento (sorteia duas posições e inverte a ordem das cidades entre elas), com 5% por novo indivíduo | Variante do swap da aula. Altera só 2 conexões da rota (o swap altera 4), então preserva o que já foi construído. A taxa é livre (`--mut-rate`): com inversão, 5% a 60% chegaram ao ótimo do círculo. Taxa 0 estaciona cedo por falta de variedade. |
| **Parada** | O primeiro entre: máximo de épocas (3000), 300 épocas sem melhora (paciência) ou ótimo conhecido atingido | Combina os critérios vistos em aula: quantidade de épocas, variação do ganho entre épocas e meta de score. |

### Parâmetros principais da função `algoritmo_genetico`

| Parâmetro | Padrão | Significado |
|---|---|---|
| `pts` | (obrigatório) | Coordenadas (n×2) das cidades |
| `tam_pop` | `None` | Tamanho da população. Com `None`, usa `6 × n` entre 100 e 600 |
| `max_ger` | 3000 | Máximo de épocas |
| `paciencia` | 300 | Épocas seguidas sem melhora antes de parar |
| `taxa_cruz` | 0.9 | Probabilidade de um par de pais cruzar |
| `mut_rate` | 0.05 | Probabilidade de mutação por indivíduo |
| `elite` | 2 | Melhores indivíduos copiados direto para a próxima geração |
| `k_torneio` | 5 | Tamanho do torneio |
| `otimo` | `None` | Distância ótima conhecida (círculo), usada para parar e calcular o gap |
| `seed` | 0 | Semente do gerador aleatório |
| `intervalo_impressao` | 50 | A cada quantas **épocas** imprimir o progresso (0 para não imprimir) |
| `nome` | `""` | Rótulo do cenário nas mensagens e gráficos |

### Ótimo conhecido no círculo

Para `n` pontos igualmente espaçados num círculo de raio `R`, o melhor caminho percorre o círculo em ordem, formando um polígono regular:

```
ótimo = 2 · n · R · sen(π / n)
```

Os pontos são embaralhados antes da execução, para que a ordem dos índices não entregue a resposta.

## Resultados observados

Valores obtidos com `--n 30 --seed 42`. Como o AG é estocástico, os números mudam com a semente e a máquina.

| Cenário | Distância final | Observação |
|---|---|---|
| Aleatório (30 pontos) | ~457 | Ótimo desconhecido; para na época 618 por estagnação |
| Círculo (30 pontos) | 313,585 | **Ótimo atingido** na época 264 (1,6 s) |
| Círculo (50 pontos, bônus) | 313,953 | **Ótimo atingido** na época 442 (4,2 s) |

Na exploração da taxa de mutação (`--explorar`, 3 repetições por taxa, cenário aleatório, 600 épocas):

- **Taxa 0:** o algoritmo estaciona logo no início (época 4, em torno de 1100), pois sem mutação a população perde variedade.
- **Taxas de 0.05 a 0.6:** resultados entre ~457 e ~510, bem próximos. Taxas baixas e a de 0.6 deram os melhores valores; 0.1 e 0.3 ficaram um pouco piores nas repetições.

### Por que inversão e não swap

Com o swap simples o círculo não chegava ao ótimo: gap de ~70% com 30 pontos, ~90% a 160% com 50 pontos e ~405% com 150 pontos (265 s). O resultado também variava muito com a semente e a taxa. O swap altera quatro conexões da rota de uma vez, um movimento brusco que desmancha trechos já bons. A inversão de segmento altera apenas duas conexões, e com ela o círculo chegou ao ótimo em todos os testes (5 sementes × 4 taxas, 30 pontos), e também com 150 pontos (2398 épocas, ~84 s).

### Limitações

O crossover da primeira metade gera pouca variedade. No cenário aleatório o ótimo é desconhecido, então não dá para medir o gap, e o resultado pode ficar em um ótimo local. Em testes com mais pontos o tempo por época cresce com o tamanho da população.

## Estrutura do projeto

```
.
├── tsp_ga.py        # código completo (cenários, AG, gráficos, CSV)
├── README.md
└── resultados/      # gerada na execução (PNGs, CSVs, GIFs)
```

Principais funções em `tsp_ga.py`:

| Função | Papel |
|---|---|
| `pontos_aleatorios`, `pontos_circulo` | Geram os cenários |
| `matriz_distancias`, `comprimentos` | Calculam a aptidão (distância total do ciclo) |
| `selecao_torneio` | Seleção por torneio |
| `order_crossover` | Cruzamento (Ordered Crossover) |
| `mutar` | Mutação por inversão de segmento |
| `algoritmo_genetico` | Laço principal de épocas e critérios de parada |
| `plota_resultado`, `salva_csv`, `salva_gif`, `tabela_por_epoca` | Acompanhamento das soluções |
| `explora_mutacao` | Comparação de taxas de mutação |

## Referência

Material da disciplina: *SIAM, Algoritmos Genéticos*. Prof. Fernando S. de Aguiar Neto, PUC-Campinas.