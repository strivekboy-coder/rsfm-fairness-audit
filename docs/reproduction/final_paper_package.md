# Final paper-facing package

The final paper package is a derived, CPU-only presentation layer. It does not
modify frozen experiment outputs or rerun models.

```powershell
python scripts/analysis/build_final_paper_package.py
```

The builder expects read-only CSV snapshots under
`work/final_paper_cleanup_sources/` and writes to
`outputs/final_paper_package_v1/`. It records source SHA256 hashes, retains the
full reBEN country×label candidate universe, and performs internal consistency
checks before declaring the package complete.
