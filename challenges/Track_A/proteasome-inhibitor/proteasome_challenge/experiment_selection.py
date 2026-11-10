"""
experiment_selection — PRELIMINARY (pre-CCSD) MP2 screening of the experimental
fragment panel for the QDC 2026 Proteasome challenge.

Scientific status of this module
--------------------------------
This module implements the *prospective MP2 screening procedure*.  Its output
is a **provisional experimental proposal**, recorded so that the exploratory
resource-matched 4/6/9 panel stays reproducible.  It is deliberately **not** a
final selection:

* it is generated from pre-CCSD data only (MP2 doubles amplitudes, orbital
  occupations, resource descriptors);
* it carries ``selection_stage = "preliminary_mp2"`` and
  ``authorizes_sqd = false``;
* it cannot be loaded by the final-selection loader
  (:mod:`proteasome_challenge.final_selection`), and nothing downstream
  (SQD calibration, tuning grid, promotion, EWF-SQD) accepts it.

Elevated MP2 doubles amplitudes do not independently establish multireference
character, CCSD inaccuracy, or the superiority of SQD.  The final, user-frozen
CCSD-informed selection is a separate artifact.

Exported functions
------------------
select_experimental_panel(catalog_df, ligand, ...)
    Deterministic MP2 downselection.  Returns a :class:`SelectionResult`.

save_preliminary_selection(result, path)
    Write the preliminary manifest to JSON.

load_preliminary_selection(path)
    Load and validate a preliminary manifest.  Raises :class:`ValueError` on
    schema violations or if the file claims to authorize SQD.

manifest_digest(manifest)
    Canonical SHA-256 of a manifest's content (used for provenance).

Selection procedure (``mp2_qubit_group_amplitude_ranking`` v1)
-------------------------------------------------------------
1. **Ligand filter**.
2. **Heavy-atom center filter** — fragments centred on hydrogen are excluded.
3. **Diagnostic-ok filter** — ``diagnostic_status == "ok"``.
4. **Qubit-group selection** — group by ``sqd_n_qubits``; choose the most
   populous group (ties: lowest ``sqd_n_qubits``).
5. **Amplitude ranking** — sort by descending ``mp2_max_abs_t2``
   (tie-break: ascending ``fragment_id``).
6. **Three representatives** by position (indices 0, n//2, n-1) →
   roles ``high_amplitude``, ``lower_median_amplitude``, ``low_amplitude``.
   For the current ixazomib catalog this reproduces fragments 4, 6, 9.

The roles describe **MP2 rank only** and are not scientific classes.

Schema version
--------------
``"3.0"`` — adds ``selection_stage``, ``authorizes_sqd``, ``status``,
``procedure`` and ``manifest_digest``; renames the old selection keys so that
none of them reads as a final choice:

=========================================  ===================================
``selected_fragment_ids`` (v2)             ``proposed_fragment_ids``
``high_amplitude_fragment_id`` (v2)        ``mp2_highest_amplitude_fragment_id``
=========================================  ===================================

Manifests with ``schema_version < "3.0"`` are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "SelectionResult",
    "select_experimental_panel",
    "save_preliminary_selection",
    "load_preliminary_selection",
    "manifest_digest",
    "SELECTION_STAGE_PRELIMINARY",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "3.0"
SELECTION_STAGE_PRELIMINARY = "preliminary_mp2"
_PROCEDURE_NAME = "mp2_qubit_group_amplitude_ranking"
_PROCEDURE_VERSION = "1"
_STATUS = "provisional_experimental_proposal"

_REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "selection_stage",
    "authorizes_sqd",
    "status",
    "ligand",
    "proposed_fragment_ids",
    "mp2_highest_amplitude_fragment_id",
    "panel",
    "selection_rule",
    "procedure",
    "manifest_digest",
}

# MP2-rank role labels stored in each panel entry (not scientific classes).
_ROLE_HIGH   = "high_amplitude"
_ROLE_MIDDLE = "lower_median_amplitude"
_ROLE_LOW    = "low_amplitude"


def manifest_digest(manifest: dict) -> str:
    """Canonical SHA-256 of the manifest content (excluding ``manifest_digest``)."""
    body = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Output of :func:`select_experimental_panel` (a *preliminary* proposal)."""

    ligand: str
    proposed_fragment_ids: list[int]
    mp2_highest_amplitude_fragment_id: int   # index-0 of the MP2 ranking; NOT an SQD authorization
    panel: list[dict]                        # one entry per proposed fragment
    selection_rule: str
    selected_sqd_n_qubits: int               # qubit width of the chosen group
    group_size: int                          # number of fragments in the chosen group
    catalog_provenance: dict = field(default_factory=dict)

    @property
    def n_proposed(self) -> int:
        return len(self.proposed_fragment_ids)

    def to_manifest(self) -> dict:
        """Serialisable dict suitable for saving to JSON."""
        d = asdict(self)
        d["schema_version"] = _SCHEMA_VERSION
        d["selection_stage"] = SELECTION_STAGE_PRELIMINARY
        d["authorizes_sqd"] = False
        d["status"] = _STATUS
        d["procedure"] = {
            "name": _PROCEDURE_NAME,
            "version": _PROCEDURE_VERSION,
            "inputs": "pre-CCSD only: MP2 max |t2|, sqd_n_qubits, center element, diagnostic_status",
            "note": ("Provisional experimental proposal. Not a final selection; does not "
                     "authorize SQD/TrimSQD execution."),
        }
        d["manifest_digest"] = manifest_digest(d)
        return d


# ---------------------------------------------------------------------------
# Core selection function
# ---------------------------------------------------------------------------

def select_experimental_panel(
    catalog_df: pd.DataFrame,
    ligand: str = "ixazomib",
    *,
    catalog_source: str = "challenge_data/fragment_catalog.csv",
    diagnostics_sha256: str | None = None,
) -> SelectionResult:
    """Deterministic MP2 qubit-group downselection (preliminary proposal).

    Parameters
    ----------
    catalog_df : DataFrame
        Enriched fragment catalog (as returned by
        ``Challenge.load_fragment_catalog()``) with at least columns
        ``ligand``, ``fragment_id``, ``sqd_n_qubits``, ``mp2_max_abs_t2``,
        ``center_element``, ``diagnostic_status``.
    ligand : str
        Ligand name (lower-case).
    catalog_source, diagnostics_sha256 : str
        Provenance strings recorded in the manifest (not used for I/O here).

    Raises
    ------
    ValueError
        If the enriched catalog is insufficient to produce a three-fragment
        panel.
    """
    ligand = str(ligand).lower().strip()

    # ── 1. Filter to the target ligand ─────────────────────────────────────
    sub = catalog_df[catalog_df["ligand"].str.lower().str.strip() == ligand].copy()
    if sub.empty:
        raise ValueError(f"No fragments found for ligand {ligand!r} in catalog.")

    # ── 2 & 3. Heavy-atom-center + diagnostic-ok filter ────────────────────
    _need = {"center_element", "diagnostic_status", "sqd_n_qubits", "mp2_max_abs_t2"}
    _missing = _need - set(sub.columns)
    if _missing:
        raise ValueError(
            f"Enriched catalog for {ligand} is missing columns: {sorted(_missing)}. "
            "Pass the DataFrame returned by Challenge.load_fragment_catalog() "
            "(with with_diagnostics=True and with_chemistry=True)."
        )
    ok = sub[
        (sub["diagnostic_status"] == "ok")
        & (sub["center_element"].astype(str).str.strip() != "H")
        & sub["sqd_n_qubits"].notna()
        & sub["mp2_max_abs_t2"].notna()
    ].copy()
    if ok.empty:
        raise ValueError(
            f"No {ligand} fragments pass the heavy-atom-center + diagnostic-ok + "
            "sqd_n_qubits-present filter.  Check that the catalog was loaded with "
            "with_diagnostics=True and with_chemistry=True."
        )

    ok["fragment_id"]   = ok["fragment_id"].astype(int)
    ok["sqd_n_qubits"]  = ok["sqd_n_qubits"].astype(int)

    # ── 4. Qubit-group selection ─────────────────────────────────────────────
    grp = (
        ok.groupby("sqd_n_qubits", dropna=True)
        .size()
        .reset_index(name="_count")
        .sort_values(["_count", "sqd_n_qubits"], ascending=[False, True])
    )
    target_qubits = int(grp.iloc[0]["sqd_n_qubits"])
    group_size    = int(grp.iloc[0]["_count"])

    # ── 5. Amplitude ranking within the group ───────────────────────────────
    pool = ok[ok["sqd_n_qubits"] == target_qubits].sort_values(
        ["mp2_max_abs_t2", "fragment_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    n = len(pool)
    if n < 3:
        raise ValueError(
            f"The most populous qubit group (sqd_n_qubits={target_qubits}) "
            f"has only {n} fragment(s) after filtering; at least 3 are required "
            "to select high/median/low representatives."
        )

    # ── 6. Three representatives ─────────────────────────────────────────────
    idx_high = 0
    idx_mid  = n // 2   # lower of two central positions for even n
    idx_low  = n - 1

    def _row_to_entry(row, role: str) -> dict:
        def _safe_int(v) -> int | None:
            try:
                x = float(v)
                return int(x) if not math.isnan(x) else None
            except (TypeError, ValueError):
                return None

        def _safe_float(v) -> float | None:
            try:
                x = float(v)
                return x if not math.isnan(x) else None
            except (TypeError, ValueError):
                return None

        return {
            "fragment_id":          int(row["fragment_id"]),
            "n_orb":                int(row["n_orb"]),
            "n_occ":                _safe_int(row.get("n_occ")),
            "n_vir":                _safe_int(row.get("n_vir")),
            "sqd_n_qubits":         _safe_int(row.get("sqd_n_qubits")),
            "mp2_max_abs_t2":       _safe_float(row.get("mp2_max_abs_t2")),
            "log10_fci_dimension":  (
                round(float(row["log10_fci_dimension"]), 6)
                if "log10_fci_dimension" in row.index
                and not _isnan(row.get("log10_fci_dimension")) else None
            ),
            "center_element":       str(row.get("center_element", "")),
            "chemical_environment": str(row.get("chemical_environment", ""))
                                    if "chemical_environment" in row.index else "",
            "mp2_rank_role":        role,
        }

    selected_rows = [
        _row_to_entry(pool.iloc[idx_high], _ROLE_HIGH),
        _row_to_entry(pool.iloc[idx_mid],  _ROLE_MIDDLE),
        _row_to_entry(pool.iloc[idx_low],  _ROLE_LOW),
    ]
    proposed_ids = sorted(r["fragment_id"] for r in selected_rows)
    high_id      = int(pool.iloc[idx_high]["fragment_id"])

    rule = (
        f"PRELIMINARY (MP2-only) proposal. From {ligand} fragments with "
        f"diagnostic_status='ok' and center_element!='H', choose the most populous "
        f"sqd_n_qubits group (sqd_n_qubits={target_qubits}, n={group_size}; ties "
        f"broken by lowest sqd_n_qubits).  Within the group, sort by descending "
        f"mp2_max_abs_t2 (tiebreak: lowest fragment_id) and select indices 0 "
        f"(high_amplitude), {idx_mid} (lower_median_amplitude, n//2={idx_mid}), "
        f"{idx_low} (low_amplitude)."
    )

    provenance = {"catalog_source": catalog_source}
    if diagnostics_sha256:
        provenance["diagnostics_sha256"] = diagnostics_sha256

    return SelectionResult(
        ligand=ligand,
        proposed_fragment_ids=proposed_ids,
        mp2_highest_amplitude_fragment_id=high_id,
        panel=sorted(selected_rows, key=lambda r: r["fragment_id"]),
        selection_rule=rule,
        selected_sqd_n_qubits=target_qubits,
        group_size=group_size,
        catalog_provenance=provenance,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_preliminary_selection(result: SelectionResult, path: Path | str) -> dict:
    """Write the preliminary manifest to ``path`` (JSON, UTF-8) and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = result.to_manifest()
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_preliminary_selection(path: Path | str) -> dict:
    """Load and validate a saved *preliminary* selection manifest.

    Raises :class:`ValueError` when:

    - the file does not exist;
    - ``schema_version`` is older than ``"3.0"`` (v2 manifests used keys that
      read as a final choice and carried no stage marker);
    - ``selection_stage`` is not ``"preliminary_mp2"`` (e.g. a final manifest);
    - ``authorizes_sqd`` is anything but ``False``;
    - required keys are missing, the id list is empty / non-integer, or the
      content digest does not match (the file was edited by hand).
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"Preliminary MP2 selection manifest not found: {path}\n"
            "Run Challenge.save_preliminary_selection() (Stage A) to generate it."
        )
    with path.open("r", encoding="utf-8") as fh:
        manifest: dict = json.load(fh)

    version = str(manifest.get("schema_version", "1.0"))
    if version < _SCHEMA_VERSION:
        raise ValueError(
            f"Preliminary selection manifest at {path} uses schema_version={version!r} "
            f"(current: {_SCHEMA_VERSION!r}). Older manifests used keys that read as a "
            "final choice and are not accepted. Re-run Challenge.save_preliminary_selection()."
        )
    missing = _REQUIRED_MANIFEST_KEYS - set(manifest.keys())
    if missing:
        raise ValueError(
            f"Preliminary selection manifest at {path} is missing required keys: "
            f"{sorted(missing)}"
        )
    if manifest["selection_stage"] != SELECTION_STAGE_PRELIMINARY:
        raise ValueError(
            f"Manifest at {path} has selection_stage={manifest['selection_stage']!r}; "
            f"expected {SELECTION_STAGE_PRELIMINARY!r}."
        )
    if manifest["authorizes_sqd"] is not False:
        raise ValueError(
            f"Manifest at {path} claims authorizes_sqd={manifest['authorizes_sqd']!r}. "
            "A preliminary MP2 selection can never authorize SQD execution."
        )
    if manifest_digest(manifest) != manifest["manifest_digest"]:
        raise ValueError(
            f"Preliminary selection manifest at {path} does not match its recorded "
            "manifest_digest (edited after generation). Regenerate it."
        )

    ids = manifest.get("proposed_fragment_ids")
    if not isinstance(ids, list) or len(ids) == 0:
        raise ValueError(
            f"Preliminary selection manifest has invalid proposed_fragment_ids: {ids!r}"
        )
    try:
        manifest["proposed_fragment_ids"] = [int(x) for x in ids]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"proposed_fragment_ids must be a list of ints, got {ids!r}"
        ) from exc
    manifest["mp2_highest_amplitude_fragment_id"] = int(
        manifest["mp2_highest_amplitude_fragment_id"])
    return manifest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _isnan(val: Any) -> bool:
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return True
