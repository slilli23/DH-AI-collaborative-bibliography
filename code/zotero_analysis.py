#!/usr/bin/env python3
"""
Analisi esplorativa di un export CSV da Zotero.

Produce:
  1. Distribuzione delle tipologie di item (CSV + donut Plotly)
  2. Distribuzione linguistica ibrida (campo Language + fallback su detection)
  3. Vocabolario dei tag (Manual + Automatic uniti) e matrice item-tag
     in formato long, predisposta per la successiva network analysis.

Uso:
    python zotero_analysis.py --input export.csv --outdir ./output

Dipendenze: pandas, plotly, langdetect
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from langdetect import DetectorFactory, LangDetectException, detect

# Determinismo della detection: indispensabile per la riproducibilità.
DetectorFactory.seed = 0

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

TAG_SEP = ";"

# Mappatura dei valori grezzi del campo Language verso codici ISO 639-1.
# La chiave è la stringa normalizzata (lowercase, senza spazi).
LANG_MAP = {
    "en": "en", "eng": "en", "english": "en", "en-us": "en", "en-gb": "en",
    "inglese": "en",
    "it": "it", "ita": "it", "italian": "it", "it-it": "it", "italiano": "it",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr", "fr-fr": "fr",
    "francese": "fr",
    "de": "de", "ger": "de", "deu": "de", "german": "de", "de-de": "de",
    "tedesco": "de",
    "es": "es", "spa": "es", "spanish": "es",
    "pt": "pt", "por": "pt", "portuguese": "pt",
    "nl": "nl", "dut": "nl", "nld": "nl", "dutch": "nl",
    "la": "la", "lat": "la", "latin": "la",
    "grc": "grc", "el": "el", "gre": "el", "ell": "el", "greek": "el",
}

# Etichette leggibili per la visualizzazione, in due lingue dell'interfaccia.
LANG_LABELS = {
    "it": {"en": "Inglese", "it": "Italiano", "fr": "Francese", "de": "Tedesco",
           "es": "Spagnolo", "pt": "Portoghese", "nl": "Olandese", "la": "Latino",
           "grc": "Greco antico", "el": "Greco moderno", "und": "Non determinata"},
    "en": {"en": "English", "it": "Italian", "fr": "French", "de": "German",
           "es": "Spanish", "pt": "Portuguese", "nl": "Dutch", "la": "Latin",
           "grc": "Ancient Greek", "el": "Modern Greek", "und": "Undetermined"},
}

ITEM_TYPE_LABELS = {
    "it": {
        "journalArticle": "Articolo in rivista", "book": "Monografia",
        "bookSection": "Capitolo di libro", "conferencePaper": "Contributo in atti",
        "blogPost": "Post di blog", "preprint": "Preprint", "webpage": "Pagina web",
        "report": "Report", "videoRecording": "Registrazione video", "dataset": "Dataset",
        "presentation": "Presentazione", "standard": "Standard", "document": "Documento",
        "statute": "Norma", "magazineArticle": "Articolo di magazine",
        "computerProgram": "Software", "thesis": "Tesi", "manuscript": "Manoscritto",
        "encyclopediaArticle": "Voce enciclopedica",
    },
    "en": {
        "journalArticle": "Journal article", "book": "Book",
        "bookSection": "Book chapter", "conferencePaper": "Conference paper",
        "blogPost": "Blog post", "preprint": "Preprint", "webpage": "Web page",
        "report": "Report", "videoRecording": "Video recording", "dataset": "Dataset",
        "presentation": "Presentation", "standard": "Standard", "document": "Document",
        "statute": "Statute", "magazineArticle": "Magazine article",
        "computerProgram": "Software", "thesis": "Thesis", "manuscript": "Manuscript",
        "encyclopediaArticle": "Encyclopedia entry",
    },
}

# Stringhe dei grafici.
UI = {
    "it": {
        "t_types": "Distribuzione per tipologia di item",
        "s_types": "{n} record — {k} tipologie",
        "t_langs": "Distribuzione linguistica dei contributi",
        "s_langs": "{n} record — campo Zotero con fallback su detection",
        "center": "item", "hover": "item", "other": "Altro",
        "src_field": "campo Zotero", "src_detect": "detection", "src_none": "non determinata",
    },
    "en": {
        "t_types": "Distribution by item type",
        "s_types": "{n} records — {k} types",
        "t_langs": "Language distribution of the contributions",
        "s_langs": "{n} records — Zotero field with detection fallback",
        "center": "items", "hover": "items", "other": "Other",
        "src_field": "Zotero field", "src_detect": "detection", "src_none": "undetermined",
    },
}

PALETTE = [
    "#4C6EF5", "#F59F00", "#12B886", "#E8590C", "#7048E8", "#1098AD",
    "#D6336C", "#66A80F", "#495057", "#F06595", "#3BC9DB", "#FAB005",
    "#5C7CFA", "#20C997", "#FF922B", "#845EF7", "#868E96", "#C2255C",
]


# ---------------------------------------------------------------------------
# Normalizzazione della lingua
# ---------------------------------------------------------------------------

def normalize_lang_field(value) -> str | None:
    """Riconduce il valore grezzo del campo Language a un codice ISO 639-1.

    Restituisce None se il campo è vuoto o non mappabile (es. 'und'),
    demandando la decisione al fallback.
    """
    if pd.isna(value):
        return None
    key = str(value).strip().lower().replace("_", "-")
    if key in ("", "und", "unknown", "n/a"):
        return None
    if key in LANG_MAP:
        return LANG_MAP[key]
    # Tentativo sul solo prefisso primario (es. 'pt-br' -> 'pt').
    primary = key.split("-")[0]
    return LANG_MAP.get(primary)


def build_detection_text(row: pd.Series) -> str:
    """Compone il testo su cui effettuare la detection.

    Il titolo da solo è una base statistica troppo esigua; l'abstract,
    quando disponibile, riduce sensibilmente il tasso di errore.
    """
    parts = []
    for field in ("Title", "Abstract Note"):
        val = row.get(field)
        if pd.notna(val):
            parts.append(str(val))
    text = " ".join(parts)
    # Rimozione di URL e rumore tipografico che disturbano la detection.
    text = re.sub(r"https?://\S+|doi:\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_lang(text: str) -> str:
    if len(text) < 20:  # soglia minima di affidabilità
        return "und"
    try:
        code = detect(text)
    except LangDetectException:
        return "und"
    return code.split("-")[0]


def resolve_languages(df: pd.DataFrame, ui_lang: str = "en") -> pd.DataFrame:
    ui = UI[ui_lang]
    """Aggiunge le colonne `lang` e `lang_source` con strategia ibrida."""
    langs, sources = [], []
    for _, row in df.iterrows():
        code = normalize_lang_field(row.get("Language"))
        if code:
            langs.append(code)
            sources.append(ui["src_field"])
        else:
            text = build_detection_text(row)
            code = detect_lang(text)
            langs.append(code)
            sources.append(ui["src_detect"] if code != "und" else ui["src_none"])
    out = df.copy()
    out["lang"] = langs
    out["lang_source"] = sources
    labels = LANG_LABELS[ui_lang]
    out["lang_label"] = out["lang"].map(lambda c: labels.get(c, c))
    return out


# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------

def split_tags(value) -> list[str]:
    if pd.isna(value):
        return []
    return [t.strip() for t in str(value).split(TAG_SEP) if t.strip()]


def build_tag_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unifica Manual e Automatic Tags.

    Restituisce:
      - long: una riga per coppia (item, tag), base per la network analysis;
      - freq: vocabolario con frequenza assoluta e provenienza.

    La normalizzazione è volutamente minima (trim + collasso degli spazi):
    il case folding è applicato solo per il raggruppamento, mentre la forma
    di superficie più frequente è conservata come etichetta, così da non
    perdere maiuscole significative (es. 'HTR', 'AI').
    """
    records = []
    for _, row in df.iterrows():
        manual = split_tags(row.get("Manual Tags"))
        automatic = split_tags(row.get("Automatic Tags"))
        seen = {}
        for tag in manual:
            seen[tag.lower()] = (tag, "manual")
        for tag in automatic:
            key = tag.lower()
            if key in seen:  # presente in entrambi i campi
                seen[key] = (seen[key][0], "entrambi")
            else:
                seen[key] = (tag, "automatic")
        for key, (surface, origin) in seen.items():
            records.append({
                "Key": row["Key"],
                "Title": row.get("Title"),
                "tag_key": key,
                "tag": surface,
                "origine": origin,
            })

    long = pd.DataFrame(records, columns=["Key", "Title", "tag_key", "tag", "origine"])
    if long.empty:
        return long, pd.DataFrame(columns=["tag", "frequenza", "origine"])

    freq = (
        long.groupby("tag_key")
        .agg(
            tag=("tag", lambda s: s.value_counts().index[0]),
            frequenza=("Key", "nunique"),
            origine=("origine", lambda s: "entrambi" if s.nunique() > 1 else s.iloc[0]),
        )
        .reset_index(drop=True)
        .sort_values(["frequenza", "tag"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return long, freq


# ---------------------------------------------------------------------------
# Visualizzazione
# ---------------------------------------------------------------------------

def donut(counts: pd.Series, title: str, subtitle: str, outfile: Path,
          ui: dict) -> None:
    """Genera un donut chart Plotly come HTML standalone."""
    total = int(counts.sum())
    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.55,
            sort=False,
            direction="clockwise",
            marker=dict(colors=PALETTE[: len(counts)],
                        line=dict(color="#FFFFFF", width=1.5)),
            textinfo="percent",
            texttemplate="%{percent:.1%}",
            textposition="auto",
            insidetextorientation="horizontal",
            hovertemplate="<b>%{label}</b><br>%{value} " + ui["hover"] +
                          " (%{percent:.1%})<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>{subtitle}</sup>", x=0.5, xanchor="center"),
        annotations=[dict(text=f"<b>{total}</b><br>{ui['center']}", x=0.5, y=0.5,
                          font_size=18, showarrow=False)],
        legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02),
        template="plotly_white",
        margin=dict(t=90, b=40, l=40, r=40),
        height=560,
    )
    fig.write_html(outfile, include_plotlyjs="cdn", full_html=True)


def group_minor(counts: pd.Series, min_count: int, label: str = "Altro") -> pd.Series:
    """Accorpa in una categoria residuale le classi sotto soglia."""
    if min_count <= 0:
        return counts
    major = counts[counts >= min_count]
    minor_sum = int(counts[counts < min_count].sum())
    if minor_sum:
        major = pd.concat([major, pd.Series({label: minor_sum})])
    return major


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analisi di un export Zotero (CSV).")
    ap.add_argument("--input", required=True, type=Path, help="CSV esportato da Zotero")
    ap.add_argument("--outdir", default=Path("output"), type=Path, help="Cartella di output")
    ap.add_argument("--min-count", type=int, default=0,
                    help="Accorpa in 'Altro' le tipologie sotto questa soglia (0 = disattivato)")
    ap.add_argument("--ui-lang", choices=["it", "en"], default="en",
                    help="Lingua delle etichette dei grafici")
    args = ap.parse_args()
    ui = UI[args.ui_lang]
    type_labels = ITEM_TYPE_LABELS[args.ui_lang]

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if "Publication Year" in df:  # evita la resa '2025.0' indotta dai NaN
        df["Publication Year"] = df["Publication Year"].astype("Int64")
    print(f"Caricati {len(df)} item da {args.input.name}\n")

    # --- 1. Tipologie -------------------------------------------------------
    types = df["Item Type"].fillna("non specificato")
    type_counts = types.value_counts()
    type_table = (
        type_counts.rename_axis("item_type").reset_index(name="frequenza")
        .assign(
            etichetta=lambda d: d["item_type"].map(lambda t: type_labels.get(t, t)),
            percentuale=lambda d: (d["frequenza"] / len(df) * 100).round(1),
        )[["item_type", "etichetta", "frequenza", "percentuale"]]
    )
    type_table.to_csv(args.outdir / "item_types.csv", index=False)

    plot_types = type_counts.rename(lambda t: type_labels.get(t, t))
    donut(
        group_minor(plot_types, args.min_count, ui["other"]),
        ui["t_types"],
        ui["s_types"].format(n=len(df), k=type_counts.size),
        args.outdir / "item_types.html", ui,
    )
    print("1. TIPOLOGIE")
    print(type_table.to_string(index=False), "\n")

    # --- 2. Lingue ----------------------------------------------------------
    df = resolve_languages(df, args.ui_lang)
    lang_counts = df["lang_label"].value_counts()
    lang_table = (
        lang_counts.rename_axis("lingua").reset_index(name="frequenza")
        .assign(percentuale=lambda d: (d["frequenza"] / len(df) * 100).round(1))
    )
    lang_table.to_csv(args.outdir / "languages.csv", index=False)

    donut(
        lang_counts, ui["t_langs"], ui["s_langs"].format(n=len(df)),
        args.outdir / "languages.html", ui,
    )
    print("2. LINGUE")
    print(lang_table.to_string(index=False))
    print("\n   Provenienza del dato:")
    print(df["lang_source"].value_counts().to_string(), "\n")

    # --- 3. Tag -------------------------------------------------------------
    long, freq = build_tag_table(df)
    freq.to_csv(args.outdir / "tag_vocabulary.csv", index=False)
    long.to_csv(args.outdir / "item_tags_raw.csv", index=False)

    tagged = long["Key"].nunique()
    print("3. TAG")
    print(f"   Tag distinti: {len(freq)}")
    print(f"   Occorrenze totali: {len(long)}")
    print(f"   Item con almeno un tag: {tagged}/{len(df)} ({tagged / len(df) * 100:.1f}%)")
    print(f"   Media tag per item taggato: {len(long) / tagged:.2f}" if tagged else "")
    print(f"   Hapax (frequenza 1): {(freq['frequenza'] == 1).sum()}")
    print("\n   Primi 25 tag per frequenza:")
    print(freq.head(25).to_string(index=False))

    # Tabella di join: SOLO il dato derivato, ricongiungibile all'export su `Key`.
    # Non si replicano le colonne della fonte, che resta l'unica copia autorevole.
    (df[["Key", "Title", "lang", "lang_source"]]
     .rename(columns={"lang": "language", "lang_source": "source"})
     .assign(language_label=df["lang_label"])
     [["Key", "Title", "language", "language_label", "source"]]
     .sort_values(["source", "language_label", "Title"], kind="stable")
     .to_csv(args.outdir / "language_attribution.csv", index=False, encoding="utf-8-sig"))
    print(f"\nOutput scritti in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
