# Raw data manifest

`data/raw/` is **not tracked in git** (COSMIC's academic license does not
permit redistributing exported mutation data). This file records exactly
what was downloaded so the raw inputs can be reconstructed.

## Source

[COSMIC](https://cancer.sanger.ac.uk/cosmic) (Catalogue Of Somatic Mutations
In Cancer), gene page for **TP53**, one CDS mutation export per transcript
isoform below. Requires a free COSMIC account (academic use).

**COSMIC release:** v102 (GRCh38). **Accessed:** 8 June 2025.

## Required layout

```
data/raw/
├── csvFolder/
│   ├── gene_1.csv   ... gene_19.csv
└── fastaFolder/
    ├── file_1.fasta ... file_19.fasta
```

`gene_N.csv` and `file_N.fasta` must correspond to the **same transcript**
(matched by index `N`, not by filename content) — see the table below.

## Transcript manifest (gene_N ↔ Ensembl transcript ID)

| index (N) | Ensembl transcript ID |
|---|---|
| 1  | ENST00000622645 |
| 2  | ENST00000610292 |
| 3  | ENST00000445888 |
| 4  | ENST00000620739 |
| 5  | ENST00000617185 |
| 6  | ENST00000610538 |
| 7  | ENST00000420246 |
| 8  | ENST00000619485 |
| 9  | ENST00000455263 |
| 10 | ENST00000413465 |
| 11 | ENST00000359597 |
| 12 | ENST00000615910 |
| 13 | ENST00000510385 |
| 14 | ENST00000610623 |
| 15 | ENST00000504290 |
| 16 | ENST00000619186 |
| 17 | ENST00000504937 |
| 18 | ENST00000618944 |
| 19 | canonical TP53 CDS (MANE Select / ENST00000269305); source FASTA header only says `>TP53`, not a numbered ENST ID |

## `gene_N.csv` schema (per-transcript COSMIC CDS mutation export)

Required columns, exact names:

| Column | Example | Notes |
|---|---|---|
| `Position` | `1` | CDS position, 1-indexed |
| `CDS Mutation` | `c.1dup` | HGVS-style; only `c.<pos><ref>><alt>` substitutions are used downstream, everything else (indels, `c.?`, etc.) is filtered out |
| `AA Mutation` | `p.M1?` | not used by the pipeline, kept for provenance |
| `Legacy Mutation ID` | `COSM10783226` | COSMIC ID, not used by the pipeline |
| `Count` | `1` | number of tumour samples reporting this mutation; drives instance replication |
| `Type` | `Insertion - Frameshift` | not used by the pipeline, kept for provenance |

## `file_N.fasta` schema

Single-record FASTA per transcript: header line (content not parsed by the
pipeline) followed by the transcript's CDS nucleotide sequence (ACGT only).

## Regenerating from raw data

Once `data/raw/csvFolder/` and `data/raw/fastaFolder/` are populated per the
above, run `notebooks/data_pipeline.ipynb` top to bottom. It rebuilds
`data/processed/` and `data/splits/` deterministically (neither is tracked
in git — see the top-level README).
