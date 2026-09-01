#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master_catalog.py

Builds the exoplanet-host-star catalogue for the ONC flyby kinematic
search, with provenance logging and Gaia DR3 quality flagging.

Pipeline:
1. Download the NASA Exoplanet Archive Planetary Systems (PS) table.
2. Deduplicate to unique host stars.
3. Cross-match with Gaia DR3 (direct ID -> TIC -> positional).
4. Flag Gaia quality indicators (RUWE, duplicated_source, non_single_star).
5. Cross-check RV against RAVE DR6, APOGEE DR17, GALAH DR3.
6. Record query provenance and selection funnel in manifest.json.

Usage:
    python build_master_catalog.py [--input FILE] [--output-dir DIR]
        [--gaia-search-radius ARCSEC] [--ruwe-threshold VALUE]
        [--skip-rv-crosscheck] [--limit N]

Requirements: pandas, numpy, requests, astropy, astroquery
"""

import argparse
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_master_catalog")


def _call_with_hard_timeout(func, timeout_sec, *args, **kwargs):
    """Enforce a real wall-clock timeout on a call that can block
    internally with no way to interrupt it (confirmed: Gaia.launch_job_async
    can hang for 1.5+ hours with no error, since control never returns to
    our code until the call itself resolves -- our own polling-loop
    timeouts never get a chance to run). Runs the call in a daemon
    thread and gives up after timeout_sec.

    Uses threading.Thread(daemon=True) rather than ThreadPoolExecutor:
    (1) the executor's `with` block calls shutdown(wait=True) on exit,
    which blocks until the task finishes regardless of the timeout given
    to future.result() -- confirmed to hang just as badly as the
    original problem; (2) even without `with`, non-daemon worker threads
    keep the whole process alive at the very end of the script's run if
    the call never actually returns. A daemon thread is abandoned
    cleanly: the process can still exit even if this call never does."""
    result_box = {}

    def _runner():
        try:
            result_box["value"] = func(*args, **kwargs)
        except Exception as e:
            result_box["error"] = e

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"call did not return within {timeout_sec}s")
    if "error" in result_box:
        raise result_box["error"]
    return result_box["value"]


def configure_gaia():
    """Standard ESA Gaia archive settings. Upload-table joins are used
    for cross-matching rather than the CDS mirror (which fails on
    large IN(...) queries)."""
    from astroquery.gaia import Gaia

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    return Gaia


# Force string dtype: these columns can mix 19-digit Gaia IDs with NaN,
# which pandas otherwise infers as float64 and corrupts via rounding.
ID_STRING_DTYPES = {"gaia_dr2_id": str, "gaia_dr3_id": str, "tic_id": str, "hostname": str, "hd_name": str}
GAIA_ID_STRING_DTYPES = {"gaia_source_id": str}

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

NASA_PS_COLUMNS = [
    "hostname", "pl_name", "ra", "dec",
    "sy_dist", "sy_disterr1", "sy_disterr2",
    "st_mass", "st_masserr1", "st_masserr2",
    "st_age", "st_ageerr1", "st_ageerr2",
    "tic_id", "hd_name", "gaia_dr2_id", "gaia_dr3_id", "default_flag",
    "sy_plx", "sy_plxerr1", "sy_plxerr2",
    "sy_pmra", "sy_pmraerr1", "sy_pmraerr2",
    "sy_pmdec", "sy_pmdecerr1", "sy_pmdecerr2",
    "st_radv", "st_radverr1", "st_radverr2", "st_radvlim",
]

GAIA_FIELDS = [
    "source_id", "ra", "dec",
    "parallax", "parallax_error",
    "pmra", "pmra_error",
    "pmdec", "pmdec_error",
    "radial_velocity", "radial_velocity_error",
    "phot_g_mean_mag", "ruwe", "duplicated_source", "non_single_star",
    "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
    "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
    "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr",
]

DEFAULT_RUWE_THRESHOLD = 1.4

RV_SURVEY_KEYWORDS = {
    "RAVE_DR6": "RAVE DR6 radial velocity",
    "APOGEE_DR17": "APOGEE DR17 allStar",
    "GALAH_DR3": "GALAH DR3",
}


# --------------------------------------------------------------------------
# Provenance / selection funnel
# --------------------------------------------------------------------------

class Manifest:
    """Tracks query provenance and per-stage sample sizes for the Data
    Availability statement and reproducibility appendix."""

    def __init__(self):
        self.data = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "nasa_query": None,
            "nasa_query_timestamp_utc": None,
            "nasa_input_source": None,
            "selection_funnel": [],
            "gaia_quality_thresholds": {},
            "rv_crosscheck_catalogs_resolved": {},
            "notes": [],
        }

    @classmethod
    def load_or_new(cls, path):
        """Load an existing manifest.json if present, so provenance from
        an earlier run (e.g. the original NASA query) isn't lost when a
        later run resumes from a checkpoint and never re-downloads."""
        m = cls()
        if Path(path).exists():
            try:
                m.data.update(json.loads(Path(path).read_text(encoding="utf-8")))
                log.info(f"Loaded existing manifest: {path} (carrying its provenance forward)")
            except Exception as e:
                log.warning(f"Could not load existing manifest {path}, starting fresh: {e}")
        return m

    def log_step(self, step, n, note=""):
        self.data["selection_funnel"].append({"step": step, "n_stars": int(n), "note": note})
        log.info(f"[funnel] {step}: {n}" + (f" ({note})" if note else ""))

    def note(self, text):
        self.data["notes"].append(text)

    def save(self, path):
        Path(path).write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        log.info(f"Manifest saved: {path}")


# --------------------------------------------------------------------------
# Step 1: NASA Exoplanet Archive
# --------------------------------------------------------------------------

def build_nasa_query():
    cols = ", ".join(NASA_PS_COLUMNS)
    return f"select {cols} from ps where default_flag = 1"


def fetch_nasa_ps_table(manifest, download_path, max_attempts=3):
    """Download the PS table via TAP, retrying on transient network errors."""
    query = build_nasa_query()
    timestamp = datetime.now(timezone.utc).isoformat()
    log.info(f"Querying NASA Exoplanet Archive TAP service: {query}")

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(NASA_TAP_URL, params={"query": query, "format": "csv"}, timeout=120)
            if resp.status_code != 200:
                log.error(f"TAP request failed (HTTP {resp.status_code}). Response body:\n{resp.text[:2000]}")
                resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            log.warning(f"NASA TAP request attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise
            time.sleep(15)

    Path(download_path).write_text(resp.text, encoding="utf-8")
    df = pd.read_csv(download_path, comment="#", dtype=ID_STRING_DTYPES)
    manifest.data["nasa_query"] = query
    manifest.data["nasa_query_timestamp_utc"] = timestamp
    manifest.data["nasa_input_source"] = "TAP_download"
    log.info(f"Downloaded {len(df)} planet rows -> {download_path}")
    return df



def load_nasa_ps_table(manifest, path):
    log.info(f"Loading local PS table: {path}")
    df = pd.read_csv(path, comment="#", dtype=ID_STRING_DTYPES)
    manifest.data["nasa_input_source"] = f"local_file:{path}"
    manifest.note(f"NASA PS table supplied as a local file ({path}); query date not logged for this run.")
    return df


def dedup_to_hosts(df, manifest):
    """Collapse planet rows to unique host stars."""
    if "default_flag" in df.columns:
        df = df[df["default_flag"] == 1].copy()
    hosts = df.drop_duplicates(subset="hostname", keep="first").copy()
    manifest.log_step("NASA PS -> unique hosts", len(hosts), note=f"from {len(df)} planet rows")
    return hosts


# --------------------------------------------------------------------------
# Step 2: Gaia DR3 cross-match (archive ID -> TIC -> positional)
# --------------------------------------------------------------------------

def _clean_source_id(value):
    """Extract a Gaia source_id (15+ consecutive digits) from a value
    that may include prefix text like 'Gaia DR3 ...'. Returns pd.NA
    (not np.nan): mixing np.nan into Series.apply() output coerces the
    whole result to float64, which loses precision on 19-digit IDs."""
    if pd.isna(value):
        return pd.NA
    match = re.search(r"\d{15,}", str(value))
    return int(match.group()) if match else pd.NA


def _run_async_gaia_query(Gaia, query, job_timeout=3600, poll_interval=10):
    """Submit an async TAP job and poll its phase explicitly rather than
    relying on launch_job/get_results()'s opaque internal wait."""
    job = _call_with_hard_timeout(Gaia.launch_job_async, job_timeout, query, dump_to_file=False, verbose=False)
    t0 = time.time()
    last_phase = None
    while True:
        phase = job.get_phase()
        elapsed = time.time() - t0
        if phase != last_phase:
            log.info(f"  job phase: {phase} (t={elapsed:.0f}s)")
            last_phase = phase
        if phase == "COMPLETED":
            result = job.get_results().to_pandas()
            result.columns = [c.lower() for c in result.columns]  # Gaia TAP returns upper-case names
            return result
        if phase in ("ERROR", "ABORTED"):
            raise RuntimeError(f"Gaia job ended in phase {phase}")
        if elapsed > job_timeout:
            job.abort()
            raise TimeoutError(f"Gaia job exceeded {job_timeout}s (last phase: {phase})")
        time.sleep(poll_interval)


def gaia_query_by_id(source_ids, batch_size=500, max_attempts=2):
    """Cross-match a list of Gaia source_id values against
    gaiadr3.gaia_source using batched WHERE source_id IN (...) queries,
    submitted as genuine async TAP jobs with explicit phase polling (see
    _run_async_gaia_query)."""
    Gaia = configure_gaia()

    clean_ids = [int(x) for x in source_ids if pd.notna(x)]
    if not clean_ids:
        return pd.DataFrame(columns=GAIA_FIELDS)

    cols = ", ".join(GAIA_FIELDS)
    n_batches = (len(clean_ids) + batch_size - 1) // batch_size
    frames = []
    for i in range(0, len(clean_ids), batch_size):
        batch = clean_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        ids_str = ",".join(str(x) for x in batch)
        query = f"SELECT {cols} FROM gaiadr3.gaia_source WHERE source_id IN ({ids_str})"

        for attempt in range(1, max_attempts + 1):
            log.info(f"Gaia batch {batch_num}/{n_batches} ({len(batch)} IDs), attempt {attempt}/{max_attempts}")
            try:
                result = _run_async_gaia_query(Gaia, query)
                if "source_id" in result.columns:
                    frames.append(result)
                break
            except Exception as e:
                log.warning(f"Batch {batch_num}/{n_batches} attempt {attempt}/{max_attempts} failed: {e}")
                if attempt == max_attempts:
                    log.error(f"Batch {batch_num}/{n_batches} permanently failed, skipping.")
                else:
                    time.sleep(30)

    if not frames:
        return pd.DataFrame(columns=GAIA_FIELDS)
    return pd.concat(frames, ignore_index=True)



def gaia_query_by_position(ra, dec, radius_arcsec):
    """Cone search for a single star. Used only as a per-star fallback
    when the bulk positional match (below) fails for a given batch."""
    Gaia = configure_gaia()
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    try:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        result = Gaia.cone_search_async(coord, u.Quantity(radius_arcsec, u.arcsec)).get_results()
        if len(result) == 0:
            return None
        result.sort("phot_g_mean_mag")  # brightest match is the most likely counterpart
        colname_by_lower = {c.lower(): c for c in result.colnames}
        return {c: result[0][colname_by_lower[c]] for c in GAIA_FIELDS if c in colname_by_lower}
    except Exception as e:
        log.warning(f"Positional search failed at ({ra}, {dec}): {e}")
        return None


GAIA_DR3_VIZIER_TABLE = "I/355/gaiadr3"

# Best-effort mapping from VizieR's Gaia DR3 column names to this
# script's GAIA_FIELDS names. Not verified against a live query.
# Correlation coefficients are NOT available via this table (VizieR's
# Gaia DR3 mirror doesn't carry them) -- an accepted gap, since this path
# is only used for the ~7% of stars that couldn't be resolved by the
# direct-ID or TIC methods, which do return full covariance data.
VIZIER_GAIA_COLUMN_MAP = {
    "Source": "source_id", "RA_ICRS": "ra", "DE_ICRS": "dec",
    "Plx": "parallax", "e_Plx": "parallax_error",
    "pmRA": "pmra", "e_pmRA": "pmra_error",
    "pmDE": "pmdec", "e_pmDE": "pmdec_error",
    "RV": "radial_velocity", "e_RV": "radial_velocity_error",
    "RUWE": "ruwe", "Dup": "duplicated_source", "NSS": "non_single_star",
    "Gmag": "phot_g_mean_mag",
}


def gaia_query_by_position_bulk(stars, radius_arcsec, batch_size=200, on_batch_done=None):
    """Bulk positional cross-match via CDS XMatch against Gaia DR3
    (I/355/gaiadr3), instead of an ESA upload-table spatial join (which
    failed consistently with connection resets on a real run). Reuses
    the same XMatch mechanism already working for the RV cross-check.

    stars: DataFrame with columns 'local_id', 'ra', 'dec'.
    Returns: dict local_id -> matched row (dict of GAIA_FIELDS).
    Column names are mapped via VIZIER_GAIA_COLUMN_MAP; correlation
    coefficients are not available through this path (see above)."""
    from astropy.table import Table as AstropyTable
    from astroquery.xmatch import XMatch
    import astropy.units as u

    results = {}
    stars = stars.reset_index(drop=True)
    n_batches = (len(stars) + batch_size - 1) // batch_size
    for b in range(0, len(stars), batch_size):
        batch = stars.iloc[b:b + batch_size]
        batch_num = b // batch_size + 1
        log.info(f"Positional XMatch batch {batch_num}/{n_batches} ({len(batch)} stars)")

        cat1 = AstropyTable.from_pandas(batch[["local_id", "ra", "dec"]])
        try:
            result = XMatch.query(
                cat1=cat1, cat2=f"vizier:{GAIA_DR3_VIZIER_TABLE}",
                max_distance=radius_arcsec * u.arcsec, colRA1="ra", colDec1="dec",
            ).to_pandas()
        except Exception as e:
            log.warning(f"XMatch batch {batch_num}/{n_batches} failed ({e}); "
                        f"falling back to one-by-one for this batch only.")
            for _, star in batch.iterrows():
                row = gaia_query_by_position(star["ra"], star["dec"], radius_arcsec)
                if row:
                    results[int(star["local_id"])] = row
            if on_batch_done is not None:
                on_batch_done(results)
            continue

        if result.empty or "local_id" not in result.columns:
            log.info(f"Positional XMatch batch {batch_num}/{n_batches}: 0 matches")
            if on_batch_done is not None:
                on_batch_done(results)
            continue

        result = result.rename(columns=VIZIER_GAIA_COLUMN_MAP)
        if "angDist" in result.columns:
            result = result.sort_values("angDist")  # nearest match per star
        result = result.drop_duplicates(subset="local_id", keep="first")
        # to_dict, not iterrows: iterrows forces one dtype per row,
        # which would corrupt a 19-digit source_id when mixed with floats.
        for record in result.to_dict("records"):
            results[int(record["local_id"])] = {c: record[c] for c in GAIA_FIELDS if c in record}
        log.info(f"Positional XMatch batch {batch_num}/{n_batches}: {len(result)} matches")

        if on_batch_done is not None:
            on_batch_done(results)

    return results


def _safe_gaia_id_from_tic(gaia_val):
    """Extract an int Gaia source_id from a TIC 'GAIA' field value.
    If MAST returns it as float64, precision may already be lost
    upstream; returns (value_or_None, was_float) so callers can warn."""
    if isinstance(gaia_val, (int, np.integer)):
        return int(gaia_val), False
    if isinstance(gaia_val, (float, np.floating)):
        if np.isnan(gaia_val):
            return None, False
        return int(gaia_val), True
    match = re.search(r"\d{15,}", str(gaia_val))
    return (int(match.group()), False) if match else (None, False)


def tic_to_gaia_map(tic_ids, batch_size=200):
    """Resolve Gaia DR3 source_id via TIC for stars without a direct
    archive identifier. Tries a batched query per chunk; falls back to
    one-by-one for any chunk that fails."""
    from astroquery.mast import Catalogs

    tic_ids = [t for t in tic_ids if pd.notna(t)]
    id_to_clean = {t: str(t).replace("TIC", "").strip() for t in tic_ids}

    mapping, n_no_match, n_errors = {}, 0, 0
    n_batches = (len(tic_ids) + batch_size - 1) // batch_size

    for b in range(0, len(tic_ids), batch_size):
        batch = tic_ids[b:b + batch_size]
        batch_num = b // batch_size + 1
        log.info(f"TIC->Gaia batch {batch_num}/{n_batches} ({len(batch)} IDs)")
        clean_batch = [id_to_clean[t] for t in batch]

        batch_result = None
        try:
            batch_result = Catalogs.query_criteria(catalog="Tic", ID=clean_batch)
        except Exception as e:
            log.warning(f"Batch TIC query failed ({e}); falling back to one-by-one for this batch.")

        if batch_result is not None and len(batch_result) > 0:
            found_ids = set(str(x) for x in batch_result["ID"])
            by_id = {str(row["ID"]): row for row in batch_result}
            for tic_id in batch:
                clean_id = id_to_clean[tic_id]
                if clean_id not in found_ids:
                    n_no_match += 1
                    continue
                gaia_val = by_id[clean_id]["GAIA"]
                if gaia_val is np.ma.masked or pd.isna(gaia_val) or str(gaia_val).strip() in ("", "--"):
                    n_no_match += 1
                    continue
                sid, was_float = _safe_gaia_id_from_tic(gaia_val)
                if sid is None:
                    n_no_match += 1
                    continue
                if was_float:
                    log.warning(f"TIC 'GAIA' value for {tic_id} arrived as float64 ({gaia_val!r}) -- "
                                f"the 19-digit ID may already be imprecise upstream in MAST's table. "
                                f"Verify this specific star against the Gaia archive directly.")
                mapping[tic_id] = sid
        else:
            # Fallback: one-by-one for this batch only.
            for tic_id in batch:
                try:
                    result = Catalogs.query_criteria(catalog="Tic", ID=id_to_clean[tic_id])
                    if len(result) == 0:
                        n_no_match += 1
                        continue
                    gaia_val = result[0]["GAIA"]
                    if gaia_val is np.ma.masked or pd.isna(gaia_val) or str(gaia_val).strip() in ("", "--"):
                        n_no_match += 1
                        continue
                    sid, was_float = _safe_gaia_id_from_tic(gaia_val)
                    if sid is None:
                        n_no_match += 1
                        continue
                    if was_float:
                        log.warning(f"TIC 'GAIA' value for {tic_id} arrived as float64 ({gaia_val!r}) -- "
                                    f"the 19-digit ID may already be imprecise upstream in MAST's table.")
                    mapping[tic_id] = sid
                except Exception as e:
                    n_errors += 1
                    log.warning(f"TIC->Gaia lookup failed for {tic_id}: {e}")

    log.info(f"TIC->Gaia: {len(mapping)} resolved, {n_no_match} without a Gaia counterpart, {n_errors} errors")
    return mapping


def enrich_with_gaia(hosts, manifest, radius_arcsec=2.0, checkpoint_path=None, resume=False):
    """Attach Gaia DR3 astrometry, quality indicators, and correlation
    coefficients, resolving source_id via direct ID -> TIC -> position.
    Checkpoints to disk after each stage if checkpoint_path is given.
    If resume=True, 'hosts' already has gaia_* columns from a previous
    checkpoint, and only the positional stage runs."""
    df = hosts.copy()
    if not resume:
        for col in GAIA_FIELDS:
            df[f"gaia_{col}"] = None  # object dtype: bool/int Gaia fields break float64 columns
        df["gaia_match_method"] = None  # provenance: direct_archive_id / tic_crossmatch / positional_*

    def _fill(idx, row):
        for col in GAIA_FIELDS:
            if col in row:
                try:
                    df.loc[idx, f"gaia_{col}"] = row[col]
                except Exception as e:
                    log.warning(f"Could not set gaia_{col} for row {idx}: {e}")

    def _dedup_by_source_id(gaia_df):
        """Duplicate source_id rows (e.g. two archive entries resolving to
        the same Gaia star) break .loc[sid] lookups (a Series is returned
        instead of a single row). Keep the first occurrence."""
        if "source_id" not in gaia_df.columns:
            return gaia_df
        n_before = len(gaia_df)
        gaia_df = gaia_df.drop_duplicates(subset="source_id", keep="first")
        if len(gaia_df) < n_before:
            log.warning(f"Dropped {n_before - len(gaia_df)} duplicate source_id rows from Gaia results.")
        return gaia_df

    def _checkpoint(stage_name):
        if checkpoint_path is not None:
            df.to_csv(checkpoint_path, index=False)
            log.info(f"Checkpoint saved after '{stage_name}': {checkpoint_path}")

    n_direct = n_direct_ok = n_tic = n_tic_ok = 0
    if resume:
        log.info("Resuming from checkpoint: skipping direct-ID and TIC stages (already reflected in the loaded data).")
    else:
        # (a) Direct lookup via archive-provided gaia_dr3_id
        if "gaia_dr3_id" in df.columns:
            # Int64 (not float64): .apply() returning pd.NA + int stays
            # object dtype, but astype("Int64") locks in precision safely.
            df["_sid_clean"] = df["gaia_dr3_id"].apply(_clean_source_id).astype("Int64")
            mask = df["_sid_clean"].notna()
            n_direct = int(mask.sum())
            if n_direct > 0:
                gaia_rows = _dedup_by_source_id(gaia_query_by_id(df.loc[mask, "_sid_clean"])).set_index("source_id", drop=False)
                for idx in df.index[mask]:
                    sid = df.loc[idx, "_sid_clean"]
                    if sid in gaia_rows.index:
                        _fill(idx, gaia_rows.loc[[sid]].to_dict("records")[0])
                        df.loc[idx, "gaia_match_method"] = "direct_archive_id"
                        n_direct_ok += 1
            df.drop(columns=["_sid_clean"], inplace=True)
        log.info(f"Direct archive Gaia ID: {n_direct_ok}/{n_direct} resolved")
        _checkpoint("direct archive Gaia ID")

        # (b) TIC -> Gaia for the rest
        unresolved = df["gaia_source_id"].isna()
        has_tic = unresolved & df.get("tic_id", pd.Series(False, index=df.index)).notna()
        n_tic, n_tic_ok = int(has_tic.sum()), 0
        if n_tic > 0:
            tic_map = tic_to_gaia_map(df.loc[has_tic, "tic_id"].tolist())
            if tic_map:
                gaia_rows = _dedup_by_source_id(gaia_query_by_id(list(tic_map.values()))).set_index("source_id", drop=False)
                for idx in df.index[has_tic]:
                    sid = tic_map.get(df.loc[idx, "tic_id"])
                    if sid in gaia_rows.index:
                        _fill(idx, gaia_rows.loc[[sid]].to_dict("records")[0])
                        df.loc[idx, "gaia_match_method"] = "tic_crossmatch"
                        n_tic_ok += 1
        log.info(f"TIC->Gaia: {n_tic_ok}/{n_tic} resolved")
        _checkpoint("TIC->Gaia")

    # (c) Positional cross-match for the remainder (bulk, not per-star)
    unresolved = df["gaia_source_id"].isna() & df["ra"].notna() & df["dec"].notna()
    to_search = df.index[unresolved]
    n_pos_ok = 0
    if len(to_search) > 0:
        local_id_to_idx = {local_id: idx for local_id, idx in enumerate(to_search)}
        stars = pd.DataFrame({
            "local_id": list(local_id_to_idx.keys()),
            "ra": df.loc[to_search, "ra"].values,
            "dec": df.loc[to_search, "dec"].values,
        })

        filled_so_far = set()

        def _on_batch_done(results_so_far):
            for local_id, row in results_so_far.items():
                if local_id in filled_so_far:
                    continue
                _fill(local_id_to_idx[local_id], row)
                df.loc[local_id_to_idx[local_id], "gaia_match_method"] = f"positional_{radius_arcsec}arcsec"
                filled_so_far.add(local_id)
            _checkpoint(f"positional search ({len(filled_so_far)} resolved so far)")

        matches = gaia_query_by_position_bulk(stars, radius_arcsec, on_batch_done=_on_batch_done)
        n_pos_ok = len(matches)
    log.info(f"Positional search: {n_pos_ok}/{len(to_search)} resolved")

    n_total = int(df["gaia_source_id"].notna().sum())
    method_counts = df["gaia_match_method"].value_counts().to_dict()
    manifest.log_step(
        "Gaia DR3 cross-match", n_total,
        note=f"of {len(df)} hosts; cumulative breakdown by method: {method_counts} "
             f"(this run resolved {n_direct_ok} direct ID, {n_tic_ok} via TIC, {n_pos_ok} positional)",
    )

    # Explicit flag, not inferred from gaia_match_method: the per-star
    # cone-search fallback (used when an XMatch batch itself fails)
    # queries the real ESA gaia_source table and DOES get correlation
    # coefficients, while the normal bulk XMatch path (VizieR mirror)
    # does not -- so "positional_*" alone doesn't tell you which. This
    # checks the actual columns instead, so a downstream orbit-
    # integration script can switch between full-covariance and
    # independent-Gaussian error sampling per star correctly.
    corr_cols = [f"gaia_{c}" for c in GAIA_FIELDS if c.endswith("_corr")]
    corr_present = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in corr_cols})
    df["gaia_has_covariance"] = corr_present.notna().all(axis=1)

    return df


# --------------------------------------------------------------------------
# Step 3: Gaia astrometric quality flags
# --------------------------------------------------------------------------

def _flag_is_true(value):
    """Robustly interpret a 'flag' value that may arrive as a real Python
    bool, a numpy bool/int, or a string (e.g. 'True'/'False'/'1'/'0')
    after a CSV round-trip, which does not preserve dtypes. A naive
    bool(value) on the string 'False' would incorrectly evaluate to True
    (any non-empty string is truthy in Python)."""
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "1.0")


def apply_gaia_quality_flags(df, manifest, ruwe_threshold=DEFAULT_RUWE_THRESHOLD):
    df = df.copy()
    df["qflag_ruwe_ok"] = pd.to_numeric(df["gaia_ruwe"], errors="coerce") < ruwe_threshold
    df["qflag_not_duplicated"] = ~df["gaia_duplicated_source"].apply(_flag_is_true)
    df["qflag_single_star"] = pd.to_numeric(df["gaia_non_single_star"], errors="coerce").fillna(0) == 0
    df["qflag_all_ok"] = df["qflag_ruwe_ok"].fillna(False) & df["qflag_not_duplicated"] & df["qflag_single_star"]

    manifest.data["gaia_quality_thresholds"] = {
        "ruwe_threshold": ruwe_threshold,
        "duplicated_source": "flagged if True",
        "non_single_star": "flagged if != 0",
    }
    manifest.log_step(
        "Gaia quality flags computed", len(df),
        note=f"{int(df['qflag_all_ok'].sum())}/{len(df)} pass RUWE<{ruwe_threshold}, "
             "not duplicated_source, non_single_star == 0",
    )
    return df


# --------------------------------------------------------------------------
# Step 4: radial-velocity cross-check against literature surveys
# --------------------------------------------------------------------------

# Verified catalogue identifiers (looked up independently, not via
# Vizier.find_catalogs -- see DEVELOPMENT_NOTES at the end of this file
# for why that automatic search is not used).
KNOWN_VIZIER_IDS = {
    "RAVE_DR6": "III/283",           # Steinmetz et al. 2020, RAVE DR6
    "APOGEE_DR17": "III/286",        # Abdurro'uf et al. 2022, APOGEE-2 DR17 allStar
    "GALAH_DR3": "J/MNRAS/506/150",  # Buder et al. 2021, GALAH+ DR3
}

# RV value/error columns confirmed from real XMatch output. Not guessed
# for LAMOST/Gaia-ESO -- add only after checking real data.
RV_VALUE_COLUMN_BY_SURVEY = {
    "RAVE_DR6": "HRV",
    "APOGEE_DR17": "HRV",
    "GALAH_DR3": "RVgalah",
}

RV_ERROR_COLUMN_BY_SURVEY = {
    "RAVE_DR6": "e_HRV",
    "APOGEE_DR17": "e_HRV",
}

# Fallback threshold (km/s) when a combined formal error isn't
# available. Placeholder, not calibrated against real RV scatter.
RV_DISCREPANCY_FALLBACK_THRESHOLD_KMS = 5.0
RV_DISCREPANCY_SIGMA = 3.0


def resolve_vizier_catalog(keyword, known_id=None):
    """Return a known, verified catalogue ID if given. Keyword search is
    disabled otherwise -- it was tested and found to return unrelated
    catalogues (not just a wrong data release), so an unresolved survey
    is skipped rather than silently mislabelled. See DEVELOPMENT NOTES."""
    if known_id:
        log.info(f"'{keyword}' -> {known_id} (verified identifier, not searched)")
        return known_id

    log.warning(f"No verified catalogue ID for '{keyword}' -- skipping this survey.")
    return None


def resolve_vizier_table(catalog_id):
    """XMatch needs a specific table ID in slash form (e.g.
    'II/246/out'), not a catalogue folder ID (e.g. 'II/246').
    Vizier.get_catalogs() returns table IDs in underscore form (e.g.
    'III_283_ravedr6'); this reconstructs the slash form XMatch expects.
    Uses the first table found -- see DEVELOPMENT NOTES for why."""
    from astroquery.vizier import Vizier

    try:
        tables = Vizier.get_catalogs(catalog_id)
    except Exception as e:
        log.warning(f"Could not list tables inside '{catalog_id}': {e}")
        return catalog_id

    if not tables:
        log.warning(f"No tables found inside '{catalog_id}'; using the folder ID as-is.")
        return catalog_id

    raw_ids = [t.meta.get("ID") or t.meta.get("name") for t in tables]
    if not any(raw_ids):
        log.warning(f"Could not find a table identifier in metadata for '{catalog_id}'. "
                    f"Raw metadata: {dict(tables[0].meta)}.")
        return catalog_id
    log.info(f"Tables found inside '{catalog_id}': {raw_ids}")

    prefix = catalog_id.replace("/", "_") + "_"

    def _to_slash_form(raw_id):
        suffix = raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id
        return f"{catalog_id}/{suffix}", suffix

    # First table by default -- works for RAVE/APOGEE; picking a table
    # by name (e.g. one containing "rv") backfired for GALAH, whose
    # dedicated RV table lacks its own RA/Dec columns.
    chosen_slash, chosen_suffix = _to_slash_form(raw_ids[0])
    log.info(f"Using table '{chosen_slash}' for '{catalog_id}'")
    return chosen_slash


def crossmatch_rv_survey(df, vizier_id, radius_arcsec=2.0):
    from astroquery.xmatch import XMatch
    from astropy.table import Table as AstropyTable
    import astropy.units as u

    # Prefer Gaia coordinates over archive ra/dec where available.
    ra = df["gaia_ra"].where(df["gaia_ra"].notna(), df["ra"]) if "gaia_ra" in df.columns else df["ra"]
    dec = df["gaia_dec"].where(df["gaia_dec"].notna(), df["dec"]) if "gaia_dec" in df.columns else df["dec"]
    small = pd.DataFrame({
        "hostname": df["hostname"],
        "ra": pd.to_numeric(ra, errors="coerce"),
        "dec": pd.to_numeric(dec, errors="coerce"),
    }).dropna()
    try:
        # cat1 needs str/file/Table, not a raw DataFrame.
        cat1_table = AstropyTable.from_pandas(small)
        result = XMatch.query(
            cat1=cat1_table, cat2=f"vizier:{vizier_id}",
            max_distance=radius_arcsec * u.arcsec, colRA1="ra", colDec1="dec",
        )
        return result.to_pandas()
    except Exception as e:
        log.warning(f"XMatch against {vizier_id} failed: {e}")
        return pd.DataFrame()


def crosscheck_rv_all_surveys(df, manifest, radius_arcsec=2.0):
    """Cross-check RV against each survey. Only RV-relevant columns
    (value, error if known, angular distance) are kept from each
    survey's XMatch result -- surveys like APOGEE/GALAH return dozens
    of unrelated columns (chemical abundances, photometry) that have
    nothing to do with RV validation."""
    df = df.copy()
    for short_name, keyword in RV_SURVEY_KEYWORDS.items():
        cat_id = resolve_vizier_catalog(keyword, known_id=KNOWN_VIZIER_IDS.get(short_name))
        if cat_id is None:
            manifest.data["rv_crosscheck_catalogs_resolved"][short_name] = None
            continue

        table_id = resolve_vizier_table(cat_id)
        manifest.data["rv_crosscheck_catalogs_resolved"][short_name] = table_id

        matched = crossmatch_rv_survey(df, table_id, radius_arcsec)
        if matched.empty or "hostname" not in matched.columns:
            manifest.note(f"{short_name} -> vizier:{table_id}; 0 matches (or 'hostname' missing from XMatch result).")
            continue

        # Keep exactly one match per star (nearest, i.e. smallest angular
        # distance) -- a search radius can return more than one candidate.
        if "angDist" in matched.columns:
            matched = matched.sort_values("angDist")
        matched = matched.drop_duplicates(subset="hostname", keep="first")

        # Drop everything except hostname, angDist, and the confirmed
        # RV value/error columns -- not the survey's other data.
        keep_cols = ["hostname", "angDist"]
        value_col = RV_VALUE_COLUMN_BY_SURVEY.get(short_name)
        error_col = RV_ERROR_COLUMN_BY_SURVEY.get(short_name)
        if value_col and value_col in matched.columns:
            keep_cols.append(value_col)
        if error_col and error_col in matched.columns:
            keep_cols.append(error_col)
        matched = matched[[c for c in keep_cols if c in matched.columns]]

        prefixed = matched.drop(columns=["hostname"]).add_prefix(f"rv_{short_name}_")
        prefixed["hostname"] = matched["hostname"].values
        df = df.merge(prefixed, on="hostname", how="left")

        manifest.note(
            f"{short_name} -> vizier:{table_id}; {len(matched)} stars matched. "
            f"Columns: {list(prefixed.columns)}."
        )
    manifest.log_step("RV cross-check vs. literature surveys", len(df))
    return df


# --------------------------------------------------------------------------
# Step 5: RV provenance
# --------------------------------------------------------------------------

def add_rv_provenance(df):
    """Record which source each star's adopted RV came from: Gaia DR3
    preferred over the NASA archive's aggregated st_radv. Literature
    survey matches stay in their own rv_<survey>_* columns."""
    df = df.copy()
    gaia_rv = pd.to_numeric(df["gaia_radial_velocity"], errors="coerce") if "gaia_radial_velocity" in df.columns else pd.Series(np.nan, index=df.index)
    archive_rv = pd.to_numeric(df["st_radv"], errors="coerce") if "st_radv" in df.columns else pd.Series(np.nan, index=df.index)
    has_gaia_rv = gaia_rv.notna()
    has_archive_rv = archive_rv.notna()

    df["rv_primary_value"] = np.nan
    df["rv_primary_source"] = "none"

    df.loc[has_gaia_rv, "rv_primary_value"] = gaia_rv.loc[has_gaia_rv]
    df.loc[has_gaia_rv, "rv_primary_source"] = "gaia_dr3"

    archive_only = has_archive_rv & ~has_gaia_rv
    df.loc[archive_only, "rv_primary_value"] = archive_rv.loc[archive_only]
    df.loc[archive_only, "rv_primary_source"] = "nasa_archive_st_radv"

    return df


def add_rv_comparison(df):
    """Compute Gaia minus literature-survey RV and flag discrepancies
    beyond 3sigma combined error (or a flat fallback threshold). Doesn't
    exclude anything -- large differences are usually unresolved
    binaries, so the flag is cross-checked against Gaia's own
    non_single_star indicator rather than treated as a data error."""
    df = df.copy()
    gaia_rv = pd.to_numeric(df["gaia_radial_velocity"], errors="coerce") if "gaia_radial_velocity" in df.columns else pd.Series(np.nan, index=df.index)
    has_gaia_rv = gaia_rv.notna()
    gaia_rv_err = pd.to_numeric(df.get("gaia_radial_velocity_error"), errors="coerce") if "gaia_radial_velocity_error" in df.columns else pd.Series(np.nan, index=df.index)
    is_non_single = pd.to_numeric(df.get("gaia_non_single_star"), errors="coerce").fillna(0) != 0 if "gaia_non_single_star" in df.columns else pd.Series(False, index=df.index)

    for short_name, col_name in RV_VALUE_COLUMN_BY_SURVEY.items():
        full_col = f"rv_{short_name}_{col_name}"
        diff_col = f"rv_diff_gaia_minus_{short_name}"
        flag_col = f"rv_flag_discrepant_{short_name}"
        binary_col = f"rv_discrepancy_matches_non_single_star_{short_name}"
        if full_col not in df.columns:
            continue

        survey_rv = pd.to_numeric(df[full_col], errors="coerce")
        has_survey_rv = survey_rv.notna()
        both = has_gaia_rv & has_survey_rv
        df[diff_col] = np.nan
        df.loc[both, diff_col] = gaia_rv.loc[both] - survey_rv.loc[both]

        err_col_name = RV_ERROR_COLUMN_BY_SURVEY.get(short_name)
        survey_err = pd.to_numeric(df.get(f"rv_{short_name}_{err_col_name}"), errors="coerce") if err_col_name else pd.Series(np.nan, index=df.index)
        combined_err = np.sqrt(gaia_rv_err**2 + survey_err**2)
        threshold = combined_err.where(combined_err.notna() & (combined_err > 0), RV_DISCREPANCY_FALLBACK_THRESHOLD_KMS / RV_DISCREPANCY_SIGMA)

        df[flag_col] = False
        df.loc[both, flag_col] = df.loc[both, diff_col].abs() > RV_DISCREPANCY_SIGMA * threshold.loc[both]

        df[binary_col] = False
        df.loc[both, binary_col] = df.loc[both, flag_col] & is_non_single.loc[both]

        n_compared = int(both.sum())
        n_flagged = int(df.loc[both, flag_col].sum())
        n_binary_explained = int(df.loc[both, binary_col].sum())
        if n_compared > 0:
            log.info(f"RV comparison Gaia vs {short_name} ({full_col}): "
                     f"{n_compared} stars with both values; "
                     f"mean diff = {df.loc[both, diff_col].mean():.3f}, "
                     f"std = {df.loc[both, diff_col].std():.3f}; "
                     f"{n_flagged} flagged discrepant (>{RV_DISCREPANCY_SIGMA}sigma or "
                     f"fallback {RV_DISCREPANCY_FALLBACK_THRESHOLD_KMS} km/s), "
                     f"of which {n_binary_explained} already flagged non_single_star by Gaia")
        else:
            log.info(f"RV comparison Gaia vs {short_name}: 0 stars with both values.")

    return df




# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="Path to a previously downloaded PS table CSV.")
    parser.add_argument("--output-dir", default="master_catalog_output")
    parser.add_argument("--gaia-search-radius", type=float, default=2.0, help="Positional search radius [arcsec].")
    parser.add_argument("--ruwe-threshold", type=float, default=DEFAULT_RUWE_THRESHOLD)
    parser.add_argument("--skip-rv-crosscheck", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N hosts (for a quick smoke test). "
                              "Ignored when resuming from an existing checkpoint.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load_or_new(out_dir / "manifest.json")

    gaia_cache_path = out_dir / "gaia_enriched.csv"
    partial_path = out_dir / "gaia_enriched.partial.csv"

    if args.limit and (gaia_cache_path.exists() or partial_path.exists()):
        log.warning(
            f"--limit {args.limit} was given, but an existing checkpoint was found in "
            f"{out_dir} -- resuming from a checkpoint always uses its full host set and "
            f"IGNORES --limit. Use a different --output-dir for a quick subset test, e.g. "
            f"--output-dir test_run, so it does not touch this checkpoint."
        )

    if gaia_cache_path.exists():
        log.info(f"Found existing {gaia_cache_path}, reusing it instead of re-running Gaia enrichment "
                 f"(delete this file to force a recompute). NASA Exoplanet Archive is not queried this run.")
        df_gaia = pd.read_csv(gaia_cache_path, dtype=GAIA_ID_STRING_DTYPES)
        manifest.note(f"Gaia enrichment loaded from {gaia_cache_path} (not recomputed this run).")
    elif partial_path.exists():
        log.info(f"Found {partial_path} from a previous interrupted run -- resuming from it "
                 f"(direct-ID and TIC stages will be skipped; only positional cross-match continues). "
                 f"NASA Exoplanet Archive is not queried this run.")
        resumed = pd.read_csv(partial_path, dtype=GAIA_ID_STRING_DTYPES)
        manifest.note(f"Gaia enrichment resumed from checkpoint {partial_path}.")
        df_gaia = enrich_with_gaia(resumed, manifest, radius_arcsec=args.gaia_search_radius,
                                    checkpoint_path=partial_path, resume=True)
        df_gaia.to_csv(gaia_cache_path, index=False)
        partial_path.unlink()
    else:
        # Fresh start -- NASA is only queried when there's no checkpoint to resume.
        if args.input:
            df_raw = load_nasa_ps_table(manifest, args.input)
        else:
            df_raw = fetch_nasa_ps_table(manifest, out_dir / "ps_raw_download.csv")
        manifest.log_step("NASA PS raw table", len(df_raw))
        manifest.save(out_dir / "manifest.json")  # persist provenance now, before any later stage can crash
        hosts = dedup_to_hosts(df_raw, manifest)
        if args.limit:
            hosts = hosts.head(args.limit).copy()
            log.info(f"--limit {args.limit}: processing only the first {len(hosts)} hosts.")
            manifest.note(f"--limit {args.limit} applied: only a subset of hosts was processed this run.")

        checkpoint_path = out_dir / "gaia_enriched.partial.csv"
        df_gaia = enrich_with_gaia(hosts, manifest, radius_arcsec=args.gaia_search_radius,
                                    checkpoint_path=checkpoint_path)
        df_gaia.to_csv(gaia_cache_path, index=False)
        if checkpoint_path.exists():
            checkpoint_path.unlink()  # only the completed result should remain on success
        log.info(f"Gaia enrichment result saved to {gaia_cache_path} for reuse on future runs.")

    df_qual = apply_gaia_quality_flags(df_gaia, manifest, ruwe_threshold=args.ruwe_threshold)

    if args.skip_rv_crosscheck:
        df_final = df_qual
        manifest.note("RV cross-check skipped (--skip-rv-crosscheck).")
    else:
        df_final = crosscheck_rv_all_surveys(df_qual, manifest, radius_arcsec=args.gaia_search_radius)

    required = ["gaia_ra", "gaia_dec", "gaia_parallax", "gaia_pmra", "gaia_pmdec", "gaia_radial_velocity"]
    present = [c for c in required if c in df_final.columns]
    df_final["complete_6d_kinematics"] = df_final[present].notna().all(axis=1)
    manifest.log_step(
        "Complete 6D kinematics", int(df_final["complete_6d_kinematics"].sum()),
        note="flag only -- full table retained regardless",
    )

    df_final = add_rv_provenance(df_final)
    manifest.log_step(
        "RV provenance assigned", len(df_final),
        note=f"by source: {df_final['rv_primary_source'].value_counts().to_dict()}",
    )

    df_final = add_rv_comparison(df_final)

    out_csv = out_dir / "master_catalog.csv"
    df_final.to_csv(out_csv, index=False)
    manifest.save(out_dir / "manifest.json")
    log.info(f"Final catalogue: {out_csv} ({len(df_final)} stars)")

    print("\n=== SELECTION FUNNEL ===")
    for step in manifest.data["selection_funnel"]:
        print(f"  {step['step']}: {step['n_stars']}  ({step['note']})")


if __name__ == "__main__":
    main()


# ==========================================================================
# DEVELOPMENT NOTES -- not executed, kept for whoever picks this up next
# ==========================================================================
#
# LAMOST DR7 (VizieR V/156)
# -------------------------
# Vizier.get_catalogs("V/156") lists 'V_156_dr7lrs' as the main table,
# which correctly resolves to 'V/156/dr7lrs' (same slash-reconstruction
# that works for RAVE/APOGEE/GALAH). Querying it via astroquery.xmatch
# still fails with:
#     "Specify the name of the RA/Dec columns in the input table"
# despite colRA1/colDec1 being passed explicitly. Ruled out: missing
# coordinates -- the table appears in VizieR's own list of tables with a
# computed spatial footprint, which only exists for tables with real
# RA/Dec. Not tested: whether the table's size (~10.6M rows) triggers
# different behaviour in the XMatch service.
#
# Next steps if revisited:
#   1. Capture the full HTTP response body from the failed XMatch
#      request (not just the raised exception message) -- the service
#      may return a more specific reason in the body.
#   2. Try a smaller LAMOST table instead of the full 'dr7lrs' (e.g.
#      'dr7plrs', a few hundred KB per the CDS FTP listing) to test the
#      size theory cheaply.
#   3. If neither works, check the table's actual column names via
#      Vizier.query_constraints or a direct CDS query -- the RA/Dec
#      columns might have non-standard names XMatch doesn't autodetect.
#
# Gaia-ESO DR5
# ------------
# No VizieR ID could be confirmed after several targeted searches (the
# reference paper is Hourihane et al. 2023, A&A 676, but the exact
# VizieR table ID wasn't found). Vizier.find_catalogs("Gaia-ESO survey
# DR5") was tried as an automatic fallback and, on a real run, resolved
# to VizieR V/164 -- which turned out to be a photometric catalogue with
# a Kepler magnitude field (magType='Kp') and ugriz SNR columns, nothing
# resembling Gaia-ESO spectroscopy. This is why resolve_vizier_catalog()
# no longer attempts automatic keyword search at all: it isn't a
# "slightly wrong version" risk (like APOGEE DR16 vs DR17), it's a
# "confidently wrong survey" risk.
#
# To add Gaia-ESO DR5 properly: find the correct VizieR ID by hand
# (search the CDS portal directly, http://cdsarc.u-strasbg.fr, rather
# than the find_catalogs() keyword API), then add it to both
# KNOWN_VIZIER_IDS and RV_SURVEY_KEYWORDS.
