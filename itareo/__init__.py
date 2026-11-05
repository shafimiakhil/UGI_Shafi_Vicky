"""ITAREO Phase 2 scaffolding (CLAUDE.md Sections 0 and 7).

Imports Luigi's pipeline from work_ms/ (commit e3bd6ec) and never edits it. Run every module from work_ms/ as
`python -m itareo.<module>` so that Luigi's packages (training, inference, dataset_builder, preprocessing) and this
package resolve. Modules: grid, prisma_pca, patches, models, train, evaluate, aggregate, points, unmixing, d1_smoke,
zonal; Earth Engine exports in gee/.
"""
from pathlib import Path

W = Path(__file__).resolve().parents[1]          # work_ms/
PROJECT = W.parent                                 # project root
DRV = PROJECT / "_luigi_drive_2026-10-26" / "PyProj_Slums_Mapping_SAR_MS_HS"   # Luigi's Drive copy, read-only
EVIDENCE = PROJECT / "docs" / "evidence_phase2"
