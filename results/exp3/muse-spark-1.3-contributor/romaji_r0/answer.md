# run 3: explore work dir after syntax error in run 2
import os
import sys
import pathlib

print("python:", sys.version)
try:
    print("cwd:", os.getcwd())
except Exception as e:
    print("getcwd failed:", e)

try:
    print("top list:", os.listdir("."))
except Exception as e:
    print("listdir failed:", e)

for root, dirs, files in os.walk(".", topdown=True):
    # limit depth and breadth for readability
    level = root.count(os.sep)
    if level > 3:
        dirs[:] = []
        continue
    print("DIR:", root, "subdirs:", dirs[:20], "files:", files[:50])
    # stop after reasonable amount
    if len(files) > 50:
        print("  ... truncated")

# try to show any task/readme/config files if present
candidates = []
for root, dirs, files in os.walk(".", topdown=True):
    for f in files:
        lf = f.lower()
        if lf.endswith((".md", ".txt", ".json", ".py", ".yaml", ".yml", ".toml")):
            candidates.append(os.path.join(root, f))
            if len(candidates) > 30:
                break

print("candidate docs:", candidates[:30])
for p in candidates[:20]:
    try:
        sz = os.path.getsize(p)
        print("=" * 60)
        print("FILE:", p, "size:", sz)
        with open(p, "r", errors="ignore") as fh:
            txt = fh.read(4000)
        print(txt[:4000])
    except Exception as e:
        print("read failed for", p, ":", e)
