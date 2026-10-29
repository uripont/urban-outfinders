# Urban Outfinders

A search engine for ~28k fashion products from an Indian e-commerce site.
Queries are matched against product titles, descriptions and attributes,
then ranked by TF-IDF, BM25, a custom score that blends text relevance
with rating, price and discount, or word2vec embeddings. A Flask web app
serves the results, summarises them with an LLM, and tracks how people
use the search.

## Status

- [ ] Text processing and exploratory analysis of the catalogue
- [ ] Inverted index, TF-IDF ranking and evaluation (P@K, R@K, MAP, MRR, NDCG)
- [ ] BM25, custom scoring and word2vec ranking
- [ ] Web app with RAG summaries and an analytics dashboard

## Getting started

You need [uv](https://docs.astral.sh/uv/getting-started/installation/).
While the repository is private you also need the
[GitHub CLI](https://cli.github.com/), logged in with `gh auth login`,
to download the dataset.

```bash
git clone https://github.com/uripont/urban-outfinders
cd urban-outfinders
uv sync                              # installs Python 3.12 and all dependencies into .venv
uv run python scripts/fetch_data.py  # downloads the catalogue and NLTK resources
JUPYTER_PREFER_ENV_PATH=1 uv run jupyter lab   # opens the notebooks
```

`JUPYTER_PREFER_ENV_PATH=1` makes Jupyter use the project's Python kernel even
if you have a global `python3` kernel pointing at another Python (on Windows
PowerShell, run `$env:JUPYTER_PREFER_ENV_PATH=1` first). In VS Code, pick the
`.venv` interpreter as the notebook kernel instead.

`uv sync` uses the exact versions in `uv.lock`, so everyone gets the same
environment. Add dependencies with `uv add <package>` (or
`uv add --dev <package>` for tools only used in development) and commit
the updated `pyproject.toml` and `uv.lock`.

For the LLM features, copy `.env.example` to `.env` and fill in the key.

## Reports

Each part of the project has a LaTeX report in `reports/part-N/`, built
with [Tectonic](https://tectonic-typesetting.github.io/en-US/install.html)
(one binary that downloads the LaTeX packages it needs on the first build):

```bash
brew install tectonic                  # macOS; see the link above for Windows and Linux
tectonic reports/part-1/main.tex       # writes reports/part-1/main.pdf
```

Notebooks save their figures into `reports/part-N/figures/`, so the report
always shows what the code produced. Built PDFs are git-ignored; the final
version of each report is added on purpose with
`git add -f reports/part-N/main.pdf`.

## Data

`fashion_products_dataset.json` holds 28,080 products (title, description,
brand, category, product details, seller, prices, discount, rating, stock).
It is published as an asset of the `data-v1` release and is not tracked
in git. `data/validation_labels.csv` holds relevance judgements for two
reference queries.

## Project layout

```
src/fashion_search/   search engine package
scripts/              setup and maintenance scripts
notebooks/            one notebook per part, read alongside its report
reports/              LaTeX reports, one folder per part
presentations/        talks given on the topic, with their figures' source
data/                 downloaded data (git-ignored, except the labels)
```

---

Built for the Information Retrieval and Web Analytics course, 2026.
