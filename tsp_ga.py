"""
Algoritmo Genético para o Problema do Caixeiro Viajante (TSP) 2D
=================================================================

Uso:
    python tsp_ga.py                      # roda aleatório (30 pts) e círculo (30 pts)
    python tsp_ga.py --bonus 150          # + teste com 150 pontos em círculo
    python tsp_ga.py --explorar           # + comparação de taxas de mutação
    python tsp_ga.py --gif                # + salva GIF da evolução
    python tsp_ga.py --n 40 --seed 7      # muda quantidade de pontos / semente
    python tsp_ga.py --mut-rate 0.1       # muda a taxa de mutação

Saídas (pasta ./resultados): PNGs, CSV com todas as épocas e GIFs opcionais.

DECISÕES DE PROJETO
-------------------
1) REPRESENTAÇÃO DO GENE
   Cada indivíduo é uma PERMUTAÇÃO dos índices das cidades, ex.: [3,0,5,1,...].
   O cromossomo já garante que toda cidade aparece exatamente uma vez, então
   nenhuma solução inválida é gerada. O percurso é cíclico: a última cidade
   volta para a primeira (o ponto de partida é irrelevante).

2) FUNÇÃO DE APTIDÃO (conforme os slides)
   F(I) = soma das distâncias euclidianas entre cidades adjacentes no
   caminho + distância da última cidade de volta à primeira (ciclo).
   O algoritmo MINIMIZA F: o ranking ordena os indivíduos pela menor distância.
   A restrição (visitar todas as cidades, sem repetir) fica na GENÉTICA:
   como o indivíduo é uma permutação, nunca existe solução inválida.
   A distância é lida de uma matriz pré-calculada e a população inteira é
   avaliada de forma vetorizada com NumPy.

3) SELEÇÃO
   Torneio (k=5), um dos métodos vistos em aula: sorteia k indivíduos e o de
   menor distância vence e vira pai. Permite controlar a pressão seletiva (k)
   sem depender da escala da aptidão. Sobreviventes: elitismo, os melhores
   `elite` indivíduos passam direto para a próxima geração, mantendo soluções
   efetivas até serem ultrapassadas (nunca perdemos o melhor já encontrado).

4) CRUZAMENTO
   Ordered Crossover, como no slide 22: mantém a PRIMEIRA METADE de um pai
   e preenche o resto com as cidades faltantes na ordem em que aparecem no
   outro pai. Garante permutações válidas (sem cidades repetidas), o que
   crossovers de 1 ponto/uniforme não garantem. Gera 1 filho por cruzamento,
   mantendo a população constante.

5) MUTAÇÃO
   Swap (troca de posição de duas cidades), como no slide 24, aplicada
   com probabilidade `mut_rate` por novo indivíduo. Padrão 5%, o exemplo da
   aula: valores altos perturbam demais e retardam o progresso; valores
   baixos demoram a escapar de mínimos locais. Use --explorar para comparar
   taxas.

6) POPULAÇÃO E CRITÉRIO DE PARADA
   Tamanho da população cresce com o número de cidades (~ 6*n, limitado entre
   100 e 600): o espaço de busca é (n-1)!/2, então mais cidades pedem mais
   diversidade. População pequena converge cedo a um ótimo local; grande
   demais gasta tempo por geração.
   Parada: (a) máximo de gerações, OU (b) `paciencia` gerações seguidas sem
   melhora do melhor indivíduo (estagnação = convergência prática), OU
   (c) no círculo, quando atinge o ótimo conhecido (perímetro do polígono
   regular), que serve de benchmark.
"""

import argparse
import csv
import math
import os
import time

import matplotlib
matplotlib.use("Agg")  # roda sem interface gráfica
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "resultados"


# ----------------------------------------------------------------------------
# Geração dos cenários
# ----------------------------------------------------------------------------
def pontos_aleatorios(n, rng):
    """Pontos uniformemente distribuídos em [0, 100]^2."""
    return rng.uniform(0, 100, size=(n, 2))


def pontos_circulo(n, rng, raio=50.0):
    """Pontos igualmente espaçados num círculo (benchmark).
    O ótimo é conhecido: perímetro do polígono regular de n lados.
    Os pontos são embaralhados para que a ordem do índice não entregue a resposta."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([50 + raio * np.cos(ang), 50 + raio * np.sin(ang)])
    rng.shuffle(pts)
    otimo = 2 * n * raio * math.sin(math.pi / n)
    return pts, otimo


def matriz_distancias(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


# ----------------------------------------------------------------------------
# Operadores do GA
# ----------------------------------------------------------------------------
def comprimentos(pop, D):
    """Comprimento do ciclo de cada indivíduo (vetorizado)."""
    return D[pop, np.roll(pop, -1, axis=1)].sum(axis=1)


def selecao_torneio(pop, dist, k, n_sel, rng):
    """Torneio: entre k sorteados vence o de MENOR distância (melhor aptidão)."""
    idx = rng.integers(0, len(pop), size=(n_sel, k))
    vencedores = idx[np.arange(n_sel), dist[idx].argmin(axis=1)]
    return pop[vencedores]


def order_crossover(p1, p2):
    """Ordered Crossover (slide da aula): mantém a 1ª metade de p1 e completa
    com as cidades faltantes na ordem em que aparecem em p2."""
    n = len(p1)
    b = n // 2
    filho = np.full(n, -1, dtype=p1.dtype)
    filho[:b] = p1[:b]
    restantes = p2[~np.isin(p2, p1[:b])]        # faltantes, na ordem de p2
    filho[filho == -1] = restantes
    return filho


def mutar(ind, rng):
    """Swap (slide da aula): troca a posição de duas cidades."""
    i, j = rng.choice(len(ind), 2, replace=False)
    ind[i], ind[j] = ind[j], ind[i]


# ----------------------------------------------------------------------------
# Algoritmo genético
# ----------------------------------------------------------------------------
def algoritmo_genetico(pts, tam_pop=None, max_ger=3000, paciencia=300,
                       taxa_cruz=0.9, mut_rate=0.05, elite=2, k_torneio=5,
                       otimo=None, seed=0, intervalo_impressao=50, nome=""):
    rng = np.random.default_rng(seed)
    n = len(pts)
    D = matriz_distancias(pts)
    if tam_pop is None:
        tam_pop = int(np.clip(6 * n, 100, 600))

    pop = np.array([rng.permutation(n) for _ in range(tam_pop)])
    t0 = time.perf_counter()

    hist = {"ger": [], "melhor": [], "media": [], "pior": [], "tempo": []}
    melhorias = []                      # (geracao, rota) a cada vez que melhora
    melhor_global, melhor_rota = np.inf, None
    ultima_melhora, t_conv = 0, 0.0

    print(f"\n=== {nome}: n={n} pontos | população={tam_pop} | "
          f"mutação={mut_rate} (swap) | cruzamento={taxa_cruz} (OX) | elite={elite} ===")

    for ger in range(max_ger + 1):
        comp = comprimentos(pop, D)
        i_best = comp.argmin()

        if comp[i_best] < melhor_global - 1e-9:
            melhor_global = comp[i_best]
            melhor_rota = pop[i_best].copy()
            melhorias.append((ger, melhor_rota))
            ultima_melhora = ger
            t_conv = time.perf_counter() - t0

        t = time.perf_counter() - t0
        hist["ger"].append(ger)
        hist["melhor"].append(melhor_global)
        hist["media"].append(comp.mean())
        hist["pior"].append(comp.max())
        hist["tempo"].append(t)

        if intervalo_impressao and (ger % intervalo_impressao == 0):
            extra = f" | gap {100*(melhor_global/otimo-1):6.2f}%" if otimo else ""
            print(f"  época {ger:5d} | melhor {melhor_global:9.3f} | "
                  f"média {comp.mean():9.3f}{extra} | {t:6.2f}s")

        # critérios de parada
        if otimo and melhor_global <= otimo * (1 + 1e-6):
            print(f"  -> ótimo conhecido atingido na época {ger}")
            break
        if ger - ultima_melhora >= paciencia:
            print(f"  -> estagnou por {paciencia} épocas (parada na época {ger})")
            break
        if ger == max_ger:
            print("  -> máximo de épocas atingido")
            break

        # nova geração
        ordem = np.argsort(comp)
        nova = [pop[i].copy() for i in ordem[:elite]]            # elitismo
        n_filhos = tam_pop - elite
        pais = selecao_torneio(pop, comp, k_torneio, 2 * n_filhos, rng)
        for c in range(n_filhos):
            p1, p2 = pais[2 * c], pais[2 * c + 1]
            filho = order_crossover(p1, p2) if rng.random() < taxa_cruz else p1.copy()
            if rng.random() < mut_rate:
                mutar(filho, rng)
            nova.append(filho)
        pop = np.array(nova)

    tempo_total = time.perf_counter() - t0
    res = dict(pts=pts, hist=hist, melhorias=melhorias, rota=melhor_rota,
               dist=melhor_global, otimo=otimo, tempo_total=tempo_total,
               tempo_conv=t_conv, ger_conv=ultima_melhora, nome=nome,
               tam_pop=tam_pop, mut_rate=mut_rate)
    print(f"  Resultado: distância {melhor_global:.3f}"
          + (f" (ótimo {otimo:.3f}, gap {100*(melhor_global/otimo-1):.2f}%)" if otimo else "")
          + f" | convergiu na época {ultima_melhora} em {t_conv:.2f}s "
          f"(execução total {tempo_total:.2f}s)")
    return res


# ----------------------------------------------------------------------------
# Visualização e registro
# ----------------------------------------------------------------------------
def rota_na_epoca(res, ger):
    """Melhor rota conhecida até a época `ger`."""
    rota = res["melhorias"][0][1]
    for g, r in res["melhorias"]:
        if g <= ger:
            rota = r
        else:
            break
    return rota


def desenha_rota(ax, pts, rota, titulo):
    ciclo = np.append(rota, rota[0])
    ax.plot(pts[ciclo, 0], pts[ciclo, 1], "-", lw=1.2, color="tab:blue")
    ax.scatter(pts[:, 0], pts[:, 1], s=12, color="tab:red", zorder=3)
    ax.set_title(titulo, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def salva_csv(res, arq):
    h = res["hist"]
    with open(arq, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoca", "melhor", "media", "pior", "tempo_s"])
        for i in range(len(h["ger"])):
            w.writerow([h["ger"][i], f'{h["melhor"][i]:.4f}', f'{h["media"][i]:.4f}',
                        f'{h["pior"][i]:.4f}', f'{h["tempo"][i]:.4f}'])


def plota_resultado(res, arq, n_snap=6):
    """Curva de convergência + soluções em épocas selecionadas."""
    h, pts = res["hist"], res["pts"]
    ult = h["ger"][-1]
    # épocas intermediárias espaçadas (escala log mostra melhor o início)
    epocas = sorted(set(int(round(x)) for x in
                        np.unique(np.geomspace(1, max(ult, 2), n_snap - 1)))) 
    epocas = [0] + [e for e in epocas if 0 < e < ult] + [ult]

    fig = plt.figure(figsize=(3.2 * len(epocas) / 2 + 2, 8))
    cols = int(math.ceil(len(epocas) / 2))
    gs = fig.add_gridspec(3, cols, height_ratios=[1, 1, 1.1])
    for k, e in enumerate(epocas):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        desenha_rota(ax, pts, rota_na_epoca(res, e),
                     f"época {e} | dist {h['melhor'][e]:.1f}")
    ax = fig.add_subplot(gs[2, :])
    ax.plot(h["ger"], h["melhor"], label="melhor", lw=2)
    ax.plot(h["ger"], h["media"], label="média da população", alpha=.7)
    if res["otimo"]:
        ax.axhline(res["otimo"], color="g", ls="--", label=f"ótimo ({res['otimo']:.1f})")
    ax.set_xlabel("época"); ax.set_ylabel("distância total")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle(f"{res['nome']} — final: {res['dist']:.2f} | "
                 f"convergência: época {res['ger_conv']} ({res['tempo_conv']:.2f}s)")
    fig.tight_layout()
    fig.savefig(arq, dpi=110)
    plt.close(fig)


def salva_gif(res, arq, frames=60):
    from matplotlib.animation import FuncAnimation, PillowWriter
    h, pts = res["hist"], res["pts"]
    ult = h["ger"][-1]
    eps = sorted(set(np.unique(np.geomspace(1, max(ult, 2), frames).astype(int)))) + [ult]
    eps = [0] + eps
    fig, ax = plt.subplots(figsize=(5, 5))

    def upd(i):
        ax.clear()
        e = eps[i]
        desenha_rota(ax, pts, rota_na_epoca(res, e), f"{res['nome']} | época {e} | "
                                                  f"dist {h['melhor'][e]:.1f}")
    FuncAnimation(fig, upd, frames=len(eps)).save(arq, writer=PillowWriter(fps=6))
    plt.close(fig)


def tabela_por_epoca(res, passo):
    h = res["hist"]
    print(f"\n  Desempenho por época — {res['nome']} (a cada {passo})")
    print("  época | melhor     | média      | tempo(s)")
    for i in range(0, len(h["ger"]), passo):
        print(f"  {h['ger'][i]:5d} | {h['melhor'][i]:10.3f} | {h['media'][i]:10.3f} | {h['tempo'][i]:7.2f}")
    i = len(h["ger"]) - 1
    print(f"  {h['ger'][i]:5d} | {h['melhor'][i]:10.3f} | {h['media'][i]:10.3f} | {h['tempo'][i]:7.2f}  (final)")


# ----------------------------------------------------------------------------
# Exploração da taxa de mutação
# ----------------------------------------------------------------------------
def explora_mutacao(n, seed):
    rng = np.random.default_rng(seed)
    pts = pontos_aleatorios(n, rng)
    taxas = [0.0, 0.05, 0.1, 0.3, 0.6]
    fig, ax = plt.subplots(figsize=(8, 5))
    print("\n=== Exploração da taxa de mutação (cenário aleatório) ===")
    for tx in taxas:
        finais = []
        for s in range(3):  # 3 repetições por taxa
            r = algoritmo_genetico(pts, max_ger=600, paciencia=600, mut_rate=tx,
                                   seed=s, intervalo_impressao=0, nome=f"mut={tx}")
            finais.append(r["dist"])
        ax.plot(r["hist"]["ger"], r["hist"]["melhor"], label=f"mut={tx} (média final {np.mean(finais):.1f})")
    ax.set_xlabel("época"); ax.set_ylabel("melhor distância"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Efeito da taxa de mutação (última repetição de cada taxa)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exploracao_mutacao.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
def roda_cenario(nome, pts, otimo, args, arq, **kw):
    res = algoritmo_genetico(pts, otimo=otimo, seed=args.seed, nome=nome, **kw)
    passo = max(1, res["hist"]["ger"][-1] // 15)
    tabela_por_epoca(res, passo)
    salva_csv(res, os.path.join(OUT_DIR, f"{arq}.csv"))
    plota_resultado(res, os.path.join(OUT_DIR, f"{arq}.png"))
    if args.gif:
        salva_gif(res, os.path.join(OUT_DIR, f"{arq}.gif"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="pontos dos cenários base (>=8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bonus", type=int, default=0, help="n de pontos do teste bônus (círculo)")
    ap.add_argument("--explorar", action="store_true", help="compara taxas de mutação")
    ap.add_argument("--mut-rate", type=float, default=0.05, help="taxa de mutação (aula: 0.05)")
    ap.add_argument("--gif", action="store_true", help="salva GIFs da evolução")
    args = ap.parse_args()
    assert args.n >= 8, "use pelo menos 8 pontos"

    os.makedirs(OUT_DIR, exist_ok=True)
    kw = dict(mut_rate=args.mut_rate)
    rng = np.random.default_rng(args.seed)

    # Cenário 1: pontos uniformes (ótimo desconhecido)
    roda_cenario("Aleatório", pontos_aleatorios(args.n, rng), None, args, "aleatorio", **kw)

    # Cenário 2: círculo (benchmark, ótimo conhecido)
    pts, otimo = pontos_circulo(args.n, rng)
    roda_cenario("Círculo", pts, otimo, args, "circulo", **kw)

    if args.explorar:
        explora_mutacao(args.n, args.seed)

    # Bônus: muitos pontos em círculo
    if args.bonus:
        pts, otimo = pontos_circulo(args.bonus, rng)
        res = roda_cenario(f"Círculo {args.bonus} pts", pts, otimo, args, "bonus_circulo",
                           max_ger=20000, paciencia=1500, intervalo_impressao=250, **kw)
        print(f"\n[BÔNUS] {args.bonus} pontos: tempo até convergir = {res['tempo_conv']:.2f}s "
              f"(época {res['ger_conv']}), gap final = {100*(res['dist']/otimo-1):.2f}%")

    print(f"\nArquivos salvos em ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
