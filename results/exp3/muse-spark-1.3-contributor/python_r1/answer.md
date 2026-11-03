import os, sys, glob, pathlib
print("cwd:", os.getcwd())
for d in [".", "..", "../..", "../../..", "../../../.."]:
    try:
        files = sorted(os.listdir(d))
        print("--- LIST", d, "n=", len(files))
        for f in files[:200]:
            print(" ", f)
    except Exception as e:
        print("--- LIST", d, "ERR", repr(e))
print("--- glob py ---")
for pat in ["*.py", "../*.py", "../../*.py", "*.json", "../*.json"]:
    try:
        print(pat, glob.glob(pat)[:50])
    except Exception as e:
        print(pat, "ERR", e)
print("--- python ---")
print(sys.version)
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as e:
    print("numpy ERR", e)
try:
    import torch
    print("torch", torch.__version__)
except Exception as e:
    print("torch ERR", e)
