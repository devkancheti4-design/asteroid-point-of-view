"""THE PHA DECISION LAW -- specification (supply). No expression here: every lane is machine-authored by integer program synthesis.

The body (pha_kernels.py) computes certificates and probabilities. It packs them into a 9-bit SITUATION
WORD n, and the law maps n to one of ten acts. The law never sees an orbit, a magnitude or a time: only
the situation. The body executes the act and recomputes the word, and every act changes the word, so
the loop ends.

  bit  0     INV    1 = input unusable (non-finite, e < 0, q <= 0, i outside [0, 180], H missing)
  bit  1     FAINT  1 = H > 22.0 (the label's size test fails; catalog H as given)
  bits 2-4   CERT   0 OPEN        cheap bounds straddle 0.05 au (L0 <= 0.05 < U_node), no exact MOID yet
                    1 FAR         L0 > 0.05 au   (radial gap: certainly outside)
                    2 NEAR        U_node <= 0.05 au (a node pair: certainly inside)
                    3 EXACT_FAR   exact MOID > 0.05 au, certified L <= MOID <= U
                    4 EXACT_NEAR  exact MOID <= 0.05 au, certified
                    5-7 BAD       certificate failed (L > U, or MOID outside [L, U])
  bits 5-6   UNC    0 NONE  1 LINEAR (delta method valid)  2 NEEDS_SAMPLE (linearisation invalid)  3 SAMPLED
  bits 7-8   PB     0 SURE_IN (P >= 1 - alpha)  1 MIDDLE  2 SURE_OUT (P <= alpha)  3 BAD (P not finite)
             (PB is set only when UNC is LINEAR or SAMPLED; otherwise the body leaves it 0)
  bits 9-31  must be 0 (anything else is OUTSIDE -> HUMAN)

ACTS = the rung number. The numbering is the identity, so no act map is needed: act = ctz(rung word).
  0 HUMAN              invalid input, failed certificate, bad probability, or a word outside the domain
  1 NOT_PHA_SIZE       H > 22: not a PHA whatever the orbit (no orbit work done)
  2 EXACT              the cheap screen is open: run K2(b)
  3 LINEAR             no probability yet: run K4(a)
  4 SAMPLE             the linearisation is invalid: run K4(b)
  5 PHA                nominal inside (cheap or exact) and P(hazard) >= 1 - alpha
  6 NOT_PHA_GEOMETRY   nominal outside and P(hazard) <= alpha
  7 EXACT_FOR_P        a cheap certificate whose probability is not decisive: run K2(b), recompute P
  8 OBSERVE_PHA        exact MOID inside, probability not decisive: the label may flip, follow up
  9 OBSERVE_NOT        exact MOID outside, probability not decisive: the label may flip, follow up"""

FIELDS = dict(inv=(0, 1), faint=(1, 1), cert=(2, 3), unc=(5, 2), pb=(7, 2))      # name: (shift, width)
WIDTH = 9
ACTS = ["HUMAN", "NOT_PHA_SIZE", "EXACT", "LINEAR", "SAMPLE", "PHA", "NOT_PHA_GEOMETRY", "EXACT_FOR_P",
        "OBSERVE_PHA", "OBSERVE_NOT"]
CERT = dict(OPEN=0, FAR=1, NEAR=2, EXACT_FAR=3, EXACT_NEAR=4)
UNC = dict(NONE=0, LINEAR=1, NEEDS_SAMPLE=2, SAMPLED=3)
PB = dict(SURE_IN=0, MIDDLE=1, SURE_OUT=2, BAD=3)


def s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def field(n, name):
    s, w = FIELDS[name]
    return (n >> s) & ((1 << w) - 1)


def pack(inv=0, faint=0, cert=0, unc=0, pb=0):
    return inv | faint << 1 | cert << 2 | unc << 5 | pb << 7


def decide(n):
    """THE SPECIFICATION, in plain if/else, for every int32 n. Written from the PHA definition and the
    meaning of each certificate. It is not tuned and fits nothing."""
    n = s32(n)
    if n < 0 or n >> WIDTH:
        return 0
    inv, faint, cert, unc, pb = (field(n, k) for k in ("inv", "faint", "cert", "unc", "pb"))
    if inv or cert >= 5 or pb == 3:
        return 0                    # HUMAN
    if faint:
        return 1                    # NOT_PHA_SIZE
    if cert == 0:
        return 2                    # EXACT
    if unc == 0:
        return 3                    # LINEAR
    if unc == 2:
        return 4                    # SAMPLE
    if cert in (2, 4) and pb == 0:
        return 5                    # PHA
    if cert in (1, 3) and pb == 2:
        return 6                    # NOT_PHA_GEOMETRY
    if cert in (1, 2):
        return 7                    # EXACT_FOR_P
    if cert == 4:
        return 8                    # OBSERVE_PHA
    return 9                        # OBSERVE_NOT


# The ladder: the same rules as rungs, in priority order. Lowest set bit wins; rung r's act is r.
# Each rung = OR of terms; each term = AND of field predicates. FLOOR = 9.
PRED = {
    "I_HI":      ("hi", None, WIDTH),
    "INV_EQ1":   ("eq", "inv", 1),
    "CERT_GE5":  ("ge", "cert", 5),
    "PB_EQ3":    ("eq", "pb", 3),
    "FAINT_EQ1": ("eq", "faint", 1),
    "CERT_EQ0":  ("eq", "cert", 0),
    "UNC_EQ0":   ("eq", "unc", 0),
    "UNC_EQ2":   ("eq", "unc", 2),
    "CERT_IN24": ("in", "cert", (2, 4)),
    "PB_EQ0":    ("eq", "pb", 0),
    "CERT_IN13": ("in", "cert", (1, 3)),
    "PB_EQ2":    ("eq", "pb", 2),
    "CERT_IN12": ("in", "cert", (1, 2)),
    "CERT_EQ4":  ("eq", "cert", 4),
}
LADDER = [
    [["I_HI"], ["INV_EQ1"], ["CERT_GE5"], ["PB_EQ3"]],    # 0 HUMAN
    [["FAINT_EQ1"]],                                      # 1 NOT_PHA_SIZE
    [["CERT_EQ0"]],                                       # 2 EXACT
    [["UNC_EQ0"]],                                        # 3 LINEAR
    [["UNC_EQ2"]],                                        # 4 SAMPLE
    [["CERT_IN24", "PB_EQ0"]],                            # 5 PHA
    [["CERT_IN13", "PB_EQ2"]],                            # 6 NOT_PHA_GEOMETRY
    [["CERT_IN12"]],                                      # 7 EXACT_FOR_P
    [["CERT_EQ4"]],                                       # 8 OBSERVE_PHA
]
FLOOR = len(LADDER)                                       # 9 OBSERVE_NOT


def holds(name, n):
    kind, f, arg = PRED[name]
    n = s32(n)
    if kind == "hi":
        return (n >> arg) != 0
    v = field(n, f)
    if kind == "eq":
        return v == arg
    if kind == "ge":
        return v >= arg
    if kind == "in":
        return v in arg
    raise KeyError(kind)


def ladder_act(n):
    for r, terms in enumerate(LADDER):
        if any(all(holds(p, n) for p in t) for t in terms):
            return r
    return FLOOR


def domain_rows():
    """every word of the 9-bit domain, the domain's neighbours, and outside words (high bits, negatives)"""
    import random
    rnd = random.Random(20261111)
    xs = list(range(1 << WIDTH))
    xs += [s32(1 << b) for b in range(WIDTH, 32)] + [s32((1 << b) | k) for b in range(WIDTH, 32) for k in (0, 5, 511)]
    xs += [s32(rnd.getrandbits(32)) for _ in range(20000)] + [-1, -(1 << 31), (1 << 31) - 1]
    return xs


def assert_ladder():
    bad = [n for n in domain_rows() if ladder_act(n) != decide(n)]
    assert not bad, "ladder != specification on %d words, e.g. %s" % (len(bad), bad[:5])
    return len(domain_rows())


if __name__ == "__main__":
    k = assert_ladder()
    from collections import Counter
    c = Counter(decide(n) for n in range(1 << WIDTH))
    print("ladder == specification on %d words (all %d domain words + outside words)" % (k, 1 << WIDTH))
    print("acts over the 512 domain words:", {ACTS[a]: c[a] for a in sorted(c)})
