#!/usr/bin/env python3
"""Replicate verify-repo-skill/scripts/import_repo_skill.mjs digestPortableTree() and
optionally write the result into a classification.json handoff.

Usage:
  python compute_skill_digest.py <runtime-skill-dir> [--write <classification.json>]

Algorithm (must match the importer byte-for-byte):
  sha256 over every regular file (symlinks are errors), sorted by absolute path with
  JS localeCompare-like ordering (we use plain codepoint sort; paths are ASCII), skipping
  agents/openai.yaml; per file: b"file\\0" + relpath + b"\\0" + str(len) + b"\\0" + content + b"\\0".
"""
import hashlib, json, os, sys
from pathlib import Path

def digest(root: Path) -> str:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                raise SystemExit(f"symlink in runtime tree: {p}")
            files.append(p)
    files.sort(key=lambda p: str(p))
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        if rel == "agents/openai.yaml" or rel.endswith("/agents/openai.yaml"):
            continue
        content = p.read_bytes()
        h.update(f"file\0{rel}\0{len(content)}\0".encode())
        h.update(content)
        h.update(b"\0")
    return "sha256:" + h.hexdigest()

def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1]).resolve()
    value = digest(root)
    print(value)
    if "--write" in sys.argv:
        target = Path(sys.argv[sys.argv.index("--write") + 1])
        data = json.loads(target.read_text())
        data["skill_content_sha256"] = value
        target.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote skill_content_sha256 into {target}")

if __name__ == "__main__":
    main()
