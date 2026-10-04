"""Verifica que los archivos del registro coinciden con las huellas publicadas en registro/SHA256SUMS."""
import hashlib
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "registro"
ok = True
for line in (base / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split()
    actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
    print(("OK   " if actual == digest else "FALLA"), name, actual)
    ok &= actual == digest
raise SystemExit(0 if ok else 1)
