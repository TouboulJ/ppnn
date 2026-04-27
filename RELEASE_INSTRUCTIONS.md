# How to push this repo to GitHub and Zenodo

This document walks you through the steps to publish the PP-NN code on
GitHub and obtain a Zenodo DOI for the technical report.

## 1. Push to GitHub (10 min)

Fill in `[Author Name]`, `[email]`, `[handle]`, and `[0000-0000-0000-0000]`
placeholders throughout (search-and-replace works), then:

```bash
cd ppnn-repo
git init
git add .
git commit -m "Initial release: PP-NN v0.1.0"

# Create a new public repo on github.com (let's say github.com/<handle>/ppnn)
git remote add origin https://github.com/<handle>/ppnn.git
git branch -M main
git push -u origin main
```

## 2. Tag the release (2 min)

```bash
git tag -a v0.1.0 -m "PP-NN v0.1.0 - paper companion release"
git push origin v0.1.0
```

This is essential because Zenodo will track tagged releases.

## 3. Connect Zenodo (10 min)

1. Go to <https://zenodo.org/account/settings/github/>
2. Sign in with your GitHub account (you'll need to grant Zenodo access)
3. Find your `ppnn` repository in the list and toggle it ON
4. Back on GitHub, create a new release at
   `https://github.com/<handle>/ppnn/releases/new` using the v0.1.0 tag
5. Add release notes (e.g., "Initial release accompanying paper [title]")
6. Click "Publish release"
7. Within ~2 minutes, Zenodo will automatically create a DOI for your release

## 4. Upload the technical report PDF to Zenodo (5 min)

The technical report is a separate Zenodo deposit (it is *the paper*,
not the code).

1. Go to <https://zenodo.org/uploads/new/>
2. Upload `knn_pp_v12_validated.pdf`
3. Fill in:
   - Title: "Projection-Pursuit Nearest Neighbors: companion technical report"
   - Authors: [your name + ORCID]
   - Description: copy the abstract
   - Keywords: nearest neighbors, projection pursuit, phi-divergence, mixture models
   - License: CC-BY 4.0 (recommended)
   - Communities: optional but consider "stat.ME" (statistics methodology)
   - Related identifier: link to the GitHub release DOI ("is supplemented by")
4. Click "Publish" - you get a permanent DOI like `10.5281/zenodo.NNNNNNNN`

## 5. Update the paper with the DOIs

Once you have the two DOIs:
- Code DOI (from step 3, GitHub release): `10.5281/zenodo.AAAAAAA`
- Tech report DOI (from step 4, PDF upload): `10.5281/zenodo.BBBBBBB`

Update these locations in your manuscripts:
- `knn_pp_short.tex` and `knn_pp_short_springer.tex`: replace `[Zenodo DOI to be inserted]`
- `cover_letter_stat_comp_v13.tex`: same
- `README.md` (this repo): both DOIs
- `CITATION.cff`: tech report DOI

Then recompile the cover letter and short manuscript, and you're ready
to submit to *Statistics and Computing*.

## 6. Optional: arXiv submission

For greater visibility, also submit the technical report to arXiv (stat.ME):
1. Go to <https://arxiv.org/submit>
2. Upload the `knn_pp.tex` source plus `figs/` directory
3. Category: stat.ME (Statistics - Methodology), cross-list to stat.ML
4. Get your arXiv identifier (e.g., `arXiv:2604.NNNNN`)
5. Add this as another "related identifier" in the Zenodo deposit
6. Cite both arXiv and Zenodo DOIs in the paper

## Total time budget

| Task | Time |
|------|-----:|
| Search/replace placeholders | 5 min |
| GitHub init + push | 10 min |
| Tag release | 2 min |
| Zenodo GitHub integration | 10 min |
| GitHub release page | 5 min |
| Zenodo deposit for tech report | 10 min |
| Update DOIs in manuscripts | 10 min |
| arXiv submission (optional) | 30 min |
| **Total (without arXiv)** | **~1 hour** |
| **Total (with arXiv)** | **~1.5 hours** |

After this, you have:
- Public open-source code at github.com/[handle]/ppnn
- Permanent DOI for the code release
- Permanent DOI for the technical report PDF (and arXiv ID if you submitted)
- Cover letter and manuscript with citable DOIs
- Ready to submit to Stat & Comp
