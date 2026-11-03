import os
import sys
print("PY", sys.version)
print("CWD", os.getcwd())
def show_dir(p):
    print("LISTDIR_START " + p)
    try:
        lst = os.listdir(p)
        for x in sorted(lst):
            fp = os.path.join(p, x)
            try:
                isd = os.path.isdir(fp)
                sz = 0
                if not isd:
                    try:
                        sz = os.path.getsize(fp)
                    except:
                        sz = -1
                print("ENTRY " + x + " isdir=" + str(isd) + " size=" + str(sz))
            except Exception as e:
                print("ENTRY_ERR " + x + " " + str(e))
    except Exception as e:
        print("LIST_ERR " + p + " " + str(e))
    print("LISTDIR_END " + p)
cwd = "/Users/sayan/Projects/weight-space-programming/results/exp7/runs/weights_only/muse-spark-1.3-contributor/pair100-gemma/attempt4/work"
show_dir(cwd)
show_dir(os.path.dirname(cwd))
show_dir(os.path.dirname(os.path.dirname(cwd)))
show_dir(os.path.dirname(os.path.dirname(os.path.dirname(cwd))))
show_dir("/Users/sayan/Projects/weight-space-programming/results/exp7")
show_dir("/Users/sayan/Projects/weight-space-programming")
def walk_limited(top, max_files=100):
    print("WALK_START " + top)
    cnt = 0
    for root, dirs, files in os.walk(top):
        print("DIR " + root + " ndirs=" + str(len(dirs)) + " nfiles=" + str(len(files)))
        for f in sorted(files)[:50]:
            print(" FILE " + os.path.join(root, f))
            cnt += 1
            if cnt > max_files:
                print("WALK_TRUNC")
                print("WALK_END")
                return
        if cnt > max_files:
            break
        if root.count(os.sep) - top.count(os.sep) > 4:
            dirs[:] = []
    print("WALK_END")
walk_limited("/Users/sayan/Projects/weight-space-programming/results/exp7/runs/weights_only/muse-spark-1.3-contributor/pair100-gemma", 200)
def cat_try(p, limit=8000):
    print("CAT_START " + p)
    try:
        with open(p, "r", errors="ignore") as fh:
            data = fh.read(limit)
            print(data[:limit])
    except Exception as e:
        print("CAT_ERR " + str(e))
    print("CAT_END " + p)
import pathlib
pair_base = "/Users/sayan/Projects/weight-space-programming/results/exp7/runs/weights_only/muse-spark-1.3-contributor/pair100-gemma"
for root, dirs, files in os.walk(pair_base):
    for f in files:
        fp = os.path.join(root, f)
        if f.endswith(".md") or f.endswith(".txt") or f.endswith(".json") or f == "prompt" or f == "task" or f.startswith("README"):
            cat_try(fp)
for root, dirs, files in os.walk("/Users/sayan/Projects/weight-space-programming/results/exp7"):
    for f in files:
        if f.endswith(".md") and root.count(os.sep) < 12:
            fp = os.path.join(root, f)
            cat_try(fp)
    break
cat_try("/Users/sayan/Projects/weight-space-programming/results/exp7/runs/weights_only/muse-spark-1.3-contributor/pair100-gemma/attempt4/work/log.txt")
print("DONE")
