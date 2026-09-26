# Sanma corpus manifests

The raw Tenhou downloads and converted `.mjson` files are intentionally not
stored in Git. They live under `koromo/source`, `koromo/mjlog`, and
`koromo/mjson`, all of which are ignored. The JSON manifests in this directory
record the inclusive date range, file counts, byte totals, and a deterministic
SHA-256 over each corpus tree.

Recreate a window with:

```powershell
.\scripts\fetch_sanma_data.ps1 -StartDate 20260701 -EndDate 20260701 -Quiet
```

The downloader filters Tenhou's hanchan and sanma GO bits; the converter
rejects yonma files; `checks/data/sanma_data_check.py` verifies the resulting three
player JSONL boundaries and event arrays.
