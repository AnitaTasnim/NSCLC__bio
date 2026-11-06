"""Draw every figure of the paper and the supplement:  python make_all.py [names...]"""
import sys
import style as S
import data as D
import main_figs
S.setup()
FUNCS = list(main_figs.ALL)
try:
    import concept_figs; FUNCS = concept_figs.ALL + FUNCS
except ImportError:
    pass
try:
    import supp_figs; FUNCS += supp_figs.ALL
except ImportError:
    pass
want = sys.argv[1:]
for f in FUNCS:
    if want and not any(w in f.__name__ for w in want):
        continue
    print(f.__name__, "->", f(D.OUT))
