#!/usr/bin/env python3
"""
Normalizzazione del vocabolario dei tag (livelli L0-L3).

Genera una tabella di mappatura esplicita (mapping.csv) e ne applica il
contenuto, producendo un dataset item-tag riconciliato pronto per la
network analysis.

La mappatura è un artefatto editoriale versionabile: lo script NON contiene
regole hard-coded oltre a quelle dichiarate in MAPPING, e può essere
rieseguito su una mapping.csv revisionata dall'utente (--mapping).

Uso:
    python normalize_tags.py --long 3_item_tag_long.csv --outdir ./output
    python normalize_tags.py --long 3_item_tag_long.csv --mapping mapping_rivisto.csv
"""

import argparse
from pathlib import Path

import pandas as pd

KEEP, MERGE, DROP, FACET = "keep", "merge", "drop", "facet"

# ---------------------------------------------------------------------------
# MAPPATURA
#   tag originale -> (azione, concetto canonico, etichetta grafo, nota)
#   Le note segnalano le decisioni INTERPRETATIVE, da validare.
#   I tag non elencati sono mantenuti invariati (azione = keep implicito).
# ---------------------------------------------------------------------------

MAPPING: dict[str, tuple[str, str, str, str]] = {}


def rule(originals, azione, canonico="", etichetta="", nota=""):
    for o in ([originals] if isinstance(originals, str) else originals):
        MAPPING[o.lower()] = (azione, canonico, etichetta, nota)


# --- FACCETTA 'target' -----------------------------------------------------
# Vocabolario controllato dei DESTINATARI, richiesto dal protocollo di tagging
# condiviso. Non è aboutness: estratto come attributo dell'item, NON come nodo,
# perché co-occorrerebbe per costruzione con ogni descrittore tematico.
rule(["ricerca", "Ricerca"], FACET, "Research")
rule(["scuola"], FACET, "School")
rule(["tutti"], FACET, "All")
rule(["università"], FACET, "University",
     nota="Treated as an audience tag, consistently with the other three")

# --- L1: stoplist ----------------------------------------------------------
rule("no tag", DROP, nota="Placeholder, not a descriptor (85 occurrences)")
rule(["English", "French"], DROP, nota="Language tag: redundant with the Language field")
rule(["FAQ", "Manuale", "scientific report"], DROP,
     nota="Document type, redundant with Item Type")

# Metadati importati da arXiv / Crossref / editore (non indicizzazione umana)
_meta = [
    "FOS: Computer and information sciences",
    "Computer Science - Artificial Intelligence",
    "Computer Science - Computation and Language",
    "Computer Science - Human-Computer Interaction",
    "Computer Science - Computer Vision and Pattern Recognition",
    "Computer Science - Databases",
    "Computer Science - Digital Libraries",
    "Computer Science - Information Retrieval",
    "Computer Science - Machine Learning",
    "Computation and Language (cs.CL)",
    "Computers and Society (cs.CY)",
    "Machine Learning (cs.LG)",
    "Human-Computer Interaction (cs.HC)",
    "Computer Vision and Pattern Recognition (cs.CV)",
    "Computational Engineering, Finance, and Science (cs.CE)",
    "Condensed Matter - Materials Science",
    "Physics - Chemical Physics",
    "Mathematics",
    "Psychology / General",
    "CCS Concepts Human",
    "CCS Concepts Human-centered computing → Accessibility →Accessibility design and evaluation methods",
    "centered computing → Accessibility →Accessibility design and evaluation methods",
]
rule(_meta, DROP, nota="Imported arXiv/publisher metadata, not human indexing")

# --- L0 + L2: refusi, sigle, unificazione interlinguistica -----------------

# Intelligenza artificiale e modelli
rule(["AI", "Artifical intelligence"], MERGE, "Artificial intelligence", "AI")
rule(["GenAI", "Intelligenza artificiale generativa"], MERGE, "Generative artificial intelligence", "GenAI")
rule(["ML", "machine learning"], MERGE, "Machine learning", "ML")
rule(["LLM", "Large Language Models"], MERGE, "Large language models", "LLM")
rule(["NLP", "Natural Language Processing", "Natural laguage processing",
      "Natural language processing systems"], MERGE, "Natural language processing", "NLP",
     nota="Includes correction of the typo 'laguage' (5 occurrences)")
rule(["neural nets", "neural network", "Neural networks"], MERGE, "Neural networks")
rule(["Sistemi di intelligenza artificiale"], MERGE, "AI systems",
     nota="Legal term (AI Act): kept distinct from 'Artificial intelligence'")
rule(["AI models", "modelli di intelligenza artificiale"], MERGE, "AI models")
rule(["foundational model"], MERGE, "Foundation models")
rule(["General-Purpose AI (GPAI)", "GPAI Models"], MERGE, "General-purpose AI", "GPAI")
rule(["Artificial General Intelligence – AGI"], MERGE, "Artificial general intelligence", "AGI")
rule(["Artificial Narrow Intelligence – ANI"], MERGE, "Artificial narrow intelligence", "ANI")
rule(["Artificial Super Intelligence – ASI"], MERGE, "Artificial super intelligence", "ASI")
rule(["Claude", "Claude models"], MERGE, "Claude")
rule(["Retrieval Augmented Generation"], MERGE, "Retrieval-augmented generation", "RAG")
rule(["Reasoning LLMs"], KEEP, "Reasoning LLMs")
rule(["Promp engineering"], MERGE, "Prompt engineering", nota="Typo correction")
rule(["Promp injection"], MERGE, "Prompt injection", nota="Typo correction")
rule(["pappagalli stocastici"], MERGE, "Stochastic parrots")
rule(["intelligenza condivisa"], MERGE, "Shared intelligence")
rule(["Technologica humanism"], MERGE, "Technological humanism", nota="Typo correction")

# Riconoscimento del testo
rule(["HTR", "Handwritten text recognition", "handwriting recognition",
      "HTR for historical documents", "HTR for medieval French manuscripts",
      "HTR for medieval Latin manuscripts"], MERGE, "Handwritten text recognition", "HTR",
     nota="The three 'HTR for...' variants are domain specifications of the same concept")
rule(["Guidelines for HTR"], MERGE, "HTR guidelines")
rule(["OCR", "Optical character recognition OCR",
      "Reconnaissance optique de caract{\\`e}res OCR"], MERGE,
     "Optical character recognition", "OCR", nota="Includes LaTeX de-escaping")
rule(["automatic transcription", "Transcription"], MERGE, "Automatic transcription",
     nota="Kept distinct from HTR: transcription is the task, HTR the technique")

# Discipline e domini
rule(["DH"], MERGE, "Digital humanities", "DH")
rule(["DH history"], MERGE, "History of digital humanities")
rule(["Philosophy", "filosofia"], MERGE, "Philosophy")
rule(["ethics", "etica"], MERGE, "Ethics")
rule(["Narratology", "narratologia"], MERGE, "Narratology")
rule(["History", "storia", "Histoire"], MERGE, "History")
rule(["Cybernetics", "Cybernétique", "Kybernetik"], MERGE, "Cybernetics")
rule(["Philology"], KEEP, "Philology")
rule(["filologia digitale"], MERGE, "Digital philology")
rule(["letteratura"], MERGE, "Literature")
rule(["letteratura italiana"], MERGE, "Italian literature")
rule(["social science"], MERGE, "Social sciences")
rule(["sociology"], MERGE, "Sociology")
rule(["economics"], MERGE, "Economics")
rule(["humanities"], MERGE, "Humanities")
rule(["SSH"], MERGE, "Social sciences and humanities", "SSH")
rule(["Computer science"], KEEP, "Computer science")
rule(["Approche diachronique", "Diachronic Approach"], MERGE, "Diachronic approach")
rule(["Approccio multidisciplinare"], MERGE, "Multidisciplinary approach")

# Educazione
rule(["education", "educazione", "istruzione"], MERGE, "Education")
rule(["university", "higher education"], MERGE, "Higher education")
rule(["preservice teachers"], MERGE, "Preservice teachers")
rule(["history teaching"], KEEP, "History teaching")
rule(["historical thinking"], KEEP, "Historical thinking")

# Ricerca
rule(["research integrity"], KEEP, "Research integrity")
rule(["Metodo scientifico"], MERGE, "Scientific method")
rule(["Scientific and technical research"], MERGE, "Scientific research")
rule(["Scientific research workflows"], KEEP, "Scientific research workflows")
rule(["scientific progress"], KEEP, "Scientific progress")
rule(["research staff"], KEEP, "Research staff")
rule(["EU research policy"], KEEP, "EU research policy")
rule(["Riproducibilità"], MERGE, "Reproducibility")
rule(["sperimentazione"], MERGE, "Experimentation")
rule(["Systematic literature review"], KEEP, "Systematic literature review")
rule(["Methodology"], KEEP, "Methodology")
rule(["Peer review"], KEEP, "Peer review")
rule(["Scholarly publishing"], KEEP, "Scholarly publishing")
rule(["lettera aperta", "open letter"], MERGE, "Open letter")

# GLAM
rule(["biblioteche", "Libraries"], MERGE, "Libraries")
rule(["digital libraries", "Digital library"], MERGE, "Digital libraries")
rule(["Public libraries"], KEEP, "Public libraries")
rule(["Software libraries"], KEEP, "Software libraries",
     nota="Homonym of 'Libraries': software sense, do NOT merge")
rule(["archivi", "archives"], MERGE, "Archives")
rule(["musei"], MERGE, "Museums")
rule(["patrimonio culturale", "beni culturali", "cultural heritage"], MERGE, "Cultural heritage")
rule(["Digital cultural heritage"], KEEP, "Digital cultural heritage")
rule(["Conservation of cultural heritage"], KEEP, "Conservation of cultural heritage")
rule(["Valorisation of cultural heritage"], KEEP, "Valorisation of cultural heritage")
rule(["istituzioni culturali"], MERGE, "Cultural institutions")
rule(["Biblioteca Apostolica Vaticana"], KEEP, "Vatican Apostolic Library")
rule(["born-digital records"], KEEP, "Born-digital records")
rule(["digitised records"], KEEP, "Digitised records")
rule(["vincolo archivistico"], MERGE, "Archival bond")
rule(["fascicoli"], MERGE, "Archival files")
rule(["aggregazioni documentali"], MERGE, "Documentary aggregations")
rule(["digitalizzazione", "Digitisation"], MERGE, "Digitisation")
rule(["art museum labels"], KEEP, "Art museum labels")
rule(["IFLA"], KEEP, "IFLA")
rule(["AI in archives"], KEEP, "AI in archives")
rule(["AI in libraries"], KEEP, "AI in libraries")
rule(["Data librarianship"], KEEP, "Data librarianship")

# Organizzazione della conoscenza
rule(["MAB"], MERGE, "GLAM institutions", "MAB",
     nota="Treated as subject matter (MAB institutions as a topic), not as audience")
rule(["Organizzazione della conoscenza"], MERGE, "Knowledge organization")
rule(["Catalogazione", "cataloging"], MERGE, "Cataloguing")
rule(["classificazione"], MERGE, "Classification")
rule(["indicizzazione"], MERGE, "Indexing")
rule(["Metadata", "metadati"], MERGE, "Metadata")
rule(["Ontologies", "ontology"], MERGE, "Ontologies")
rule(["OWL ontology"], KEEP, "OWL ontology")
rule(["Linked Open Data"], KEEP, "Linked open data", "LOD")
rule(["Vocabulary"], KEEP, "Vocabulary")
rule(["Dewey Decimal Classification"], KEEP, "Dewey Decimal Classification", "DDC")
rule(["DDC short numbers"], KEEP, "DDC short numbers")
rule(["Library of Congress Subject Heading"], KEEP, "Library of Congress Subject Headings", "LCSH")
rule(["Library of Congress Classification"], KEEP, "Library of Congress Classification", "LCC")
rule(["DigiClass"], KEEP, "DigiClass")
rule(["Machine-based classification"], KEEP, "Machine-based classification")
rule(["Knowledge engineering"], KEEP, "Knowledge engineering")
rule(["accesso alla conoscenza"], MERGE, "Access to knowledge")

# Diritto e governance
rule(["leggi", "law", "diritto"], MERGE, "Law")
rule(["Normativa", "regolamentazione"], MERGE, "Regulation")
rule(["AI regulation"], KEEP, "AI regulation")
rule(["AI law"], KEEP, "AI law")
rule(["AI act", "EU AI Act"], MERGE, "EU AI Act")
rule(["Diritto delle Nuove Tecnologie"], MERGE, "Technology law")
rule(["dottrina giuridica su IA"], MERGE, "AI legal scholarship")
rule(["Diritto d'Autore", "Copyright"], MERGE, "Copyright")
rule(["gestione del rischio", "Risk Management"], MERGE, "Risk management")
rule(["rischi"], MERGE, "Risk")
rule(["standard ISO", "ISO Standards"], MERGE, "ISO standards")
rule(["Linee guida"], MERGE, "Guidelines")
rule(["Linee guida MIM"], MERGE, "MIM guidelines (Italy)")
rule(["Buone pratiche"], MERGE, "Best practices")
rule(["Amministrazione Pubblica", "Public administration"], MERGE, "Public administration")
rule(["uso responsabile", "sviluppo responsabile", "Responsible AI"], MERGE, "Responsible AI",
     nota="Merges 'uso responsabile' and 'sviluppo responsabile'")
rule(["affidabilità"], MERGE, "Reliability")
rule(["Inclusion", "Inclusiveness"], MERGE, "Inclusion")
rule(["DEI (Diversity, Equity, Inclusion)"], KEEP, "Diversity, equity and inclusion", "DEI")
rule(["AIMS (Artificial Intelligence Management System)"], MERGE, "AI management systems", "AIMS")
rule(["policy"], MERGE, "Policy")
rule(["policymaking"], MERGE, "Policymaking")
rule(["sostenibilità"], MERGE, "Sustainability")
rule(["lavoro"], MERGE, "Labour")
rule(["industria"], MERGE, "Industry")
rule(["Italia"], MERGE, "Italy")

# Bias, autorialità, studi letterari
rule(["Artificial intelligence authorship"], KEEP, "AI authorship")
rule(["Distribute authorship"], MERGE, "Distributed authorship", nota="Typo correction")
rule(["Suspence"], MERGE, "Suspense", nota="Typo correction")
rule(["Multilingal information retrieval", "Recherche d'information multilingue"], MERGE,
     "Multilingual information retrieval", nota="Includes correction of the typo 'Multilingal'")
rule(["discorso indiretto libero"], MERGE, "Free indirect discourse")
rule(["Tempo nella letteratura"], MERGE, "Time in literature")
rule(["Esthetic appreciation"], MERGE, "Aesthetic appreciation")
rule(["art"], MERGE, "Art")
rule(["CULTURE"], MERGE, "Culture")
rule(["desire"], MERGE, "Desire")
rule(["emotions"], MERGE, "Emotions")
rule(["communication"], MERGE, "Communication")
rule(["Indeterminatezza"], MERGE, "Indeterminacy")
rule(["Produzioni di testo"], MERGE, "Text production")
rule(["Testo", "text"], MERGE, "Text")
rule(["iconografia"], MERGE, "Iconography")
rule(["riconoscimento immagini"], MERGE, "Image recognition")
rule(["epigrafia", "Epigraphy"], MERGE, "Epigraphy")
rule(["scrittura cipro-minoica"], MERGE, "Cypro-Minoan script")
rule(["Fascismo"], MERGE, "Fascism")
rule(["Greek World", "Greek World/history"], MERGE, "Greek world")
rule(["text-to-image", "text-to-image generation"], MERGE, "Text-to-image generation")
rule(["digital editions", "digital scholarly edition"], MERGE, "Digital scholarly editions")
rule(["Medieval manuscripts", "handwritten medieval documents"], MERGE, "Medieval manuscripts")
rule(["Medieval text"], MERGE, "Medieval texts")
rule(["Moyen Âge"], MERGE, "Middle Ages")
rule(["Old prints", "Imprim{\\'e}s anciens"], MERGE, "Early printed books",
     nota="Includes LaTeX de-escaping")
rule(["Ecosistema digitale"], MERGE, "Digital ecosystem")
rule(["datasets"], MERGE, "Datasets")
rule(["benchmark", "benchmarking"], MERGE, "Benchmarking")
rule(["evaluation"], MERGE, "Evaluation")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_mapping_table(long: pd.DataFrame) -> pd.DataFrame:
    """Costruisce mapping.csv a partire dal vocabolario osservato."""
    vocab = (
        long.groupby("tag_key")
        .agg(tag_originale=("tag", lambda s: s.value_counts().index[0]),
             occorrenze=("Key", "nunique"),
             origine=("origine", lambda s: "both" if s.nunique() > 1 else s.iloc[0]))
        .reset_index()
    )
    rows = []
    for _, r in vocab.iterrows():
        azione, canonico, etichetta, nota = MAPPING.get(
            r.tag_key, (KEEP, r.tag_originale, "", ""))
        rows.append({
            "original_tag": r.tag_originale,
            "items": r.occorrenze,
            "source_field": r.origine,
            "action": azione,
            "canonical_concept": canonico if azione != DROP else "",
            "graph_label": etichetta,
            "note": nota,
        })
    return (pd.DataFrame(rows)
            .sort_values(["items", "canonical_concept"], ascending=[False, True])
            .reset_index(drop=True))


IT2EN = {"tag_originale": "original_tag", "occorrenze": "items",
         "origine": "source_field", "azione": "action",
         "concetto_canonico": "canonical_concept",
         "etichetta_grafo": "graph_label", "nota": "note"}


def harmonise(mapping: pd.DataFrame) -> pd.DataFrame:
    """Accetta anche mapping.csv con le vecchie intestazioni italiane."""
    m = mapping.rename(columns=IT2EN)
    if "source_field" in m:
        m["source_field"] = m["source_field"].replace({"entrambi": "both"})
    return m


def apply_mapping(long: pd.DataFrame, mapping: pd.DataFrame,
                  min_freq: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restituisce (concetti tematici, faccetta target).

    I tag marcati `facet` escono dal vocabolario dei nodi e confluiscono in un
    attributo a livello di item.
    """
    mapping = harmonise(mapping)
    idx = {r.original_tag.lower(): r for _, r in mapping.iterrows()}
    rows, facets = [], []
    for _, r in long.iterrows():
        m = idx.get(r.tag_key)
        if m is None or m.action == DROP:
            continue
        if m.action == FACET:
            facets.append({"Key": r.Key, "Title": r.Title,
                           "audience": m.canonical_concept, "original_tag": r.tag})
            continue
        rows.append({"Key": r.Key, "Title": r.Title,
                     "concept": m.canonical_concept,
                     "graph_label": m.graph_label or m.canonical_concept,
                     "original_tag": r.tag})

    out = pd.DataFrame(rows).drop_duplicates(subset=["Key", "concept"])  # L2: dedup post-merge
    target = (pd.DataFrame(facets, columns=["Key", "Title", "audience", "original_tag"])
              .drop_duplicates(subset=["Key", "audience"]))

    if min_freq > 1:  # L3
        keep = out.groupby("concept")["Key"].nunique()
        out = out[out.concept.isin(keep[keep >= min_freq].index)]
    return out.reset_index(drop=True), target.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True, type=Path, help="item_tags_raw.csv")
    ap.add_argument("--outdir", default=Path("output"), type=Path)
    ap.add_argument("--mapping", type=Path, help="tag_mapping.csv revisionata (opzionale)")
    ap.add_argument("--min-freq", type=int, default=2, help="Soglia L3 sui concetti")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(args.long)

    if args.mapping:
        mapping = pd.read_csv(args.mapping).fillna("")
        print(f"Mappatura revisionata caricata: {args.mapping.name}")
    else:
        mapping = build_mapping_table(long)
        mapping.to_csv(args.outdir / "tag_mapping.csv", index=False)

    norm, target = apply_mapping(long, mapping, args.min_freq)
    norm.to_csv(args.outdir / "item_concepts.csv", index=False)
    target.to_csv(args.outdir / "item_target.csv", index=False)

    # --- Report -------------------------------------------------------------
    n_facet = (harmonise(mapping).action == FACET).sum()
    n_drop = (harmonise(mapping).action == DROP).sum()
    n_merge = (harmonise(mapping).action == MERGE).sum()
    n_keep = (harmonise(mapping).action == KEEP).sum()
    flagged = harmonise(mapping).pipe(lambda m: m[m.note.fillna("").str.contains("VERIFICARE|CHECK")])

    pre_concepts = long[long.tag_key != "no tag"].tag_key.nunique()
    post_all, _ = apply_mapping(long, mapping, 1)

    print(f"\nVOCABOLARIO: {len(mapping)} tag -> keep {n_keep} | merge {n_merge} | "
          f"facet {n_facet} | drop {n_drop}")
    print(f"Concetti dopo L0-L2: {post_all.concept.nunique()} (da {pre_concepts} tag sostanziali)")
    print(f"Concetti dopo L3 (freq >= {args.min_freq}): {norm.concept.nunique()}")
    print(f"Item nel grafo: {norm.Key.nunique()} | occorrenze: {len(norm)}")

    hapax_pre = (long[long.tag_key != "no tag"].groupby("tag_key").Key.nunique() == 1).sum()
    hapax_post = (post_all.groupby("concept").Key.nunique() == 1).sum()
    print(f"Hapax: {hapax_pre} -> {hapax_post} "
          f"({hapax_post / post_all.concept.nunique() * 100:.0f}% del vocabolario)")

    sizes = norm.groupby("Key").size()
    print(f"\nTag burst residuo: max {sizes.max()} concetti/item (mediana {sizes.median():.0f})")
    print(f"\nFACCETTA TARGET: {target.Key.nunique()}/{long.Key.nunique()} item "
          f"({target.Key.nunique() / long.Key.nunique() * 100:.0f}% di copertura)")
    print(target.audience.value_counts().to_string())

    print(f"\nDECISIONI DA VALIDARE: {len(flagged)}")
    for _, r in flagged.iterrows():
        print(f"  [{r.items:>2}] {r.original_tag} -> {r.canonical_concept}\n"
              f"       {r.note}")

    print("\nPRIMI 20 CONCETTI:")
    top = norm.groupby("concept").Key.nunique().sort_values(ascending=False).head(20)
    print(top.to_string())
    print(f"\nOutput in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
