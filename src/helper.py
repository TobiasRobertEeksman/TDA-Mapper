import re


#helper for safe filenames
def _slug(s: str) -> str:
    # safe-ish folder/file name
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    return re.sub(r"[^-_.A-Za-z0-9]", "-", s)

def _fmt_float(x: float) -> str:
    # compact + filesystem friendly (replace '.' with 'p')
    return f"{x:.4g}".replace(".", "p")
