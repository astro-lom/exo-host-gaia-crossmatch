# build_master_catalog.py

Builds the exoplanet-host-star catalogue for future flyby kinematic
search, with provenance logging and Gaia DR3 quality
flagging. Tested end-to-end against the live NASA/Gaia/VizieR
services.

## What it does

1. Downloads the NASA Exoplanet Archive Planetary Systems (PS) table
   via TAP (exact query and timestamp logged). Deduplicates to unique
   host stars.
2. Cross-matches with Gaia DR3, in order: direct `gaia_dr3_id` → TIC →
   positional (CDS XMatch against Gaia DR3). The first two retrieve
   full astrometry plus correlation coefficients for covariance
   propagation; the positional path does not carry correlation
   coefficients (see `gaia_has_covariance`). `gaia_match_method`
   records which method resolved each star.
3. Flags Gaia quality indicators (RUWE, `duplicated_source`,
   `non_single_star`) — configurable RUWE threshold, default 1.4.
4. Cross-checks RV against independent spectroscopic surveys (see
   [Known limitations](#known-limitations) for which ones work).
5. Records RV provenance (`rv_primary_source`) and, where both Gaia
   and a survey have a value, the difference and a discrepancy flag.

Progress checkpoints after each stage; an interrupted run resumes
instead of restarting.

## Requirements

```
pip install pandas numpy requests astropy astroquery
```

## Usage

```bash
python build_master_catalog.py                              # full run
python build_master_catalog.py --limit 20 --output-dir test_run  # smoke test
python build_master_catalog.py --input my_ps_table.csv       # skip NASA download
python build_master_catalog.py --skip-rv-crosscheck          # skip RV step
```

Other options: `--output-dir` (default `master_catalog_output`),
`--gaia-search-radius` (arcsec, default 2.0), `--ruwe-threshold`
(default 1.4). `--help` for the full list.

`--limit` is ignored when resuming from an existing checkpoint in
`--output-dir` — use a separate directory for subset tests.

## Output

- `master_catalog.csv` — final catalogue, one row per host star.
- `manifest.json` — NASA query text and timestamp, selection funnel,
  quality-flag thresholds, resolved VizieR IDs, and notes.

### Key columns

| Column | Meaning |
|---|---|
| `gaia_source_id` | Gaia DR3 ID (string, to avoid float64 precision loss) |
| `gaia_match_method` | `direct_archive_id` / `tic_crossmatch` / `positional_<radius>arcsec` |
| `gaia_has_covariance` | `True` if the astrometric correlation coefficients are actually present for this star (direct ID/TIC matches have them; most positional matches don't) — downstream orbit-integration code should use this to decide between full-covariance and independent-Gaussian error sampling |
| `qflag_ruwe_ok`, `qflag_not_duplicated`, `qflag_single_star`, `qflag_all_ok` | Gaia quality flags |
| `gaia_*_corr` | Astrometric correlation coefficients (see `gaia_has_covariance`) |
| `rv_primary_value`, `rv_primary_source` | Adopted RV and its source |
| `rv_<SURVEY>_*` | Raw columns from the XMatch cross-match |
| `rv_diff_gaia_minus_<SURVEY>` | Gaia RV minus survey RV |
| `rv_flag_discrepant_<SURVEY>` | Difference exceeds 3σ combined error (or 5 km/s fallback) |
| `rv_discrepancy_matches_non_single_star_<SURVEY>` | Discrepancy coincides with Gaia's `non_single_star` flag |
| `complete_6d_kinematics` | Gaia parallax, PM, and RV all present |

## Known limitations

- **RV cross-check supports RAVE DR6, APOGEE DR17, and GALAH DR3.**
  LAMOST DR7 and Gaia-ESO DR5 are not implemented — see
  `DEVELOPMENT NOTES` at the end of the script for what would be
  needed to add them.
- **RV discrepancy threshold (3σ combined error, or a flat 5 km/s
  fallback when an error column isn't available) is a fixed default**,
  not tuned to any particular dataset.
- **Positional matches don't carry correlation coefficients** — see
  `gaia_has_covariance`.
- **TIC → Gaia ID precision**: MAST's `GAIA` field is handled
  defensively in case it's returned as float64 (which would lose
  precision on the 19-digit ID); a warning logs if that's detected.

**Note on 19-digit Gaia IDs:** pandas silently upcasts to float64 (losing
precision) in `Series.apply()` when mixing `NaN`/int, in
`.loc[single_label]`/`.iterrows()` when a row mixes int and float
columns, and in `pd.read_csv()` when a column mixes large ints with
blanks. This code avoids all three (`pd.NA` in `.apply()`,
`.to_dict("records")` instead of `.iterrows()`, explicit `dtype=str` on
read). Apply the same guards to any new large-integer column.

## Data sources

- NASA Exoplanet Archive (PS table), via TAP.
- Gaia DR3, via `astroquery.gaia` (`gaiadr3.gaia_source`, for direct-ID
  and TIC matches) and via VizieR `I/355/gaiadr3` through the CDS
  XMatch service (for positional matches).
- TESS Input Catalog (TIC), via `astroquery.mast`.
- RAVE DR6 (Steinmetz et al. 2020, VizieR III/283), APOGEE-2 DR17
  (Abdurro'uf et al. 2022, III/286), GALAH+ DR3 (Buder et al. 2021,
  J/MNRAS/506/150), via the CDS XMatch service.

## License and citation

MIT License (see `LICENSE`). Please cite via `CITATION.cff`.
