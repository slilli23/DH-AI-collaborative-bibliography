#!/usr/bin/env python3
"""
Network analysis della co-occorrenza di concetti nella bibliografia DH-AI.

Costruisce la proiezione concetto-concetto a partire dal grafo bipartito
item-concetto, con pesatura di Newman (2001): un item che indicizza n concetti
contribuisce a ciascuna delle C(n,2) coppie con peso 1/(n-1). La pesatura
impedisce che i record iper-indicizzati dominino la topologia, senza escluderli.

Uso:
    python network_analysis.py --long 5_item_concetto_long.csv --outdir ./output
"""

import argparse
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

SEED = 42

PALETTE = ["#4C6EF5", "#F59F00", "#12B886", "#E8590C", "#7048E8",
           "#1098AD", "#D6336C", "#66A80F", "#495057", "#F06595"]


def build_graph(long: pd.DataFrame, weighting: str = "newman") -> nx.Graph:
    """Proiezione concetto-concetto dal bipartito item-concetto."""
    G = nx.Graph()
    freq = long.groupby("concetto").Key.nunique()
    labels = long.drop_duplicates("concetto").set_index("concetto").etichetta
    for c, f in freq.items():
        G.add_node(c, frequenza=int(f), label=labels.get(c, c))

    for _, grp in long.groupby("Key"):
        concepts = sorted(grp.concetto.unique())
        n = len(concepts)
        if n < 2:
            continue
        w = 1 / (n - 1) if weighting == "newman" else 1.0
        for a, b in combinations(concepts, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
                G[a][b]["n_item"] += 1
            else:
                G.add_edge(a, b, weight=w, n_item=1)
    return G


def analyse(G: nx.Graph) -> tuple[pd.DataFrame, dict, float]:
    # Distanza = inverso del peso: co-occorrenze forti = nodi vicini.
    dist = {(u, v): 1 / d["weight"] for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, dist, "distance")

    strength = dict(G.degree(weight="weight"))
    betw = nx.betweenness_centrality(G, weight="distance", normalized=True)
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eig = {n: float("nan") for n in G}

    comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    membership = {n: i for i, c in enumerate(comms) for n in c}
    modularity = nx.community.modularity(G, comms, weight="weight")

    metrics = pd.DataFrame({
        "concetto": list(G.nodes),
        "frequenza": [G.nodes[n]["frequenza"] for n in G],
        "grado": [G.degree(n) for n in G],
        "forza": [round(strength[n], 3) for n in G],
        "betweenness": [round(betw[n], 4) for n in G],
        "eigenvector": [round(eig[n], 4) for n in G],
        "community": [membership[n] for n in G],
    }).sort_values("forza", ascending=False).reset_index(drop=True)
    return metrics, membership, modularity


def render(G: nx.Graph, metrics: pd.DataFrame, membership: dict, outfile: Path) -> None:
    net = Network(height="800px", width="100%", bgcolor="#ffffff",
                  font_color="#212529", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-8000, spring_length=180, spring_strength=0.02)

    m = metrics.set_index("concetto")
    for n in G.nodes:
        r = m.loc[n]
        net.add_node(
            n, label=G.nodes[n]["label"],
            size=10 + 3.2 * (r.frequenza ** 0.6),
            color=PALETTE[membership[n] % len(PALETTE)],
            title=(f"{n}\nItem: {r.frequenza} | grado: {r.grado}\n"
                   f"forza: {r.forza} | betweenness: {r.betweenness}\n"
                   f"community: {r.community}"),
        )
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d["weight"], width=0.5 + 2 * d["weight"],
                     title=f"{d['n_item']} item — peso {d['weight']:.2f}",
                     color="#adb5bd")
    net.write_html(str(outfile), notebook=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True, type=Path)
    ap.add_argument("--outdir", default=Path("output"), type=Path)
    ap.add_argument("--weighting", choices=["newman", "binary"], default="newman")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(args.long)
    G = build_graph(long, args.weighting)
    metrics, membership, mod = analyse(G)

    metrics.to_csv(args.outdir / "7_metriche_nodi.csv", index=False)
    nx.to_pandas_edgelist(G).to_csv(args.outdir / "7_edgelist.csv", index=False)  # Gephi-ready
    render(G, metrics, membership, args.outdir / "7_network.html")

    comp = list(nx.connected_components(G))
    giant = max(comp, key=len)
    print(f"GRAFO ({args.weighting}): {G.number_of_nodes()} nodi | "
          f"{G.number_of_edges()} archi | densità {nx.density(G):.3f}")
    print(f"Componenti: {len(comp)} | componente gigante: {len(giant)} nodi "
          f"({len(giant) / G.number_of_nodes() * 100:.0f}%)")
    print(f"Modularità (Louvain, seed={SEED}): {mod:.3f} — "
          f"{metrics.community.nunique()} community\n")

    print("TOP 12 PER FORZA")
    print(metrics.head(12).to_string(index=False))

    print("\nTOP 8 PER BETWEENNESS (concetti-ponte)")
    print(metrics.nlargest(8, "betweenness")[
        ["concetto", "frequenza", "forza", "betweenness", "community"]].to_string(index=False))

    print("\nCOMMUNITY")
    for c in sorted(metrics.community.unique()):
        sub = metrics[metrics.community == c].sort_values("forza", ascending=False)
        print(f"\n  [{c}] {len(sub)} concetti — {', '.join(sub.concetto.head(8))}"
              + (" …" if len(sub) > 8 else ""))

    print(f"\nOutput in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
