/* THE PHA DECISION LAW -- pha_law(n) maps a 9-bit situation word to one of ten acts.
   Machine-authored by integer program synthesis: every lane below is the synthesised expression,
   verbatim, each from a fresh search; the glue is masks and ORs only.

   n: bit 0 INV | bit 1 FAINT | bits 2-4 CERT | bits 5-6 UNC | bits 7-8 PB | bits 9-31 must be 0
   CERT 0 OPEN 1 FAR 2 NEAR 3 EXACT_FAR 4 EXACT_NEAR 5-7 BAD;  UNC 0 NONE 1 LINEAR 2 NEEDS_SAMPLE
   3 SAMPLED;  PB 0 SURE_IN 1 MIDDLE 2 SURE_OUT 3 BAD
   acts: 0 HUMAN 1 NOT_PHA_SIZE 2 EXACT 3 LINEAR 4 SAMPLE 5 PHA 6 NOT_PHA_GEOMETRY 7 EXACT_FOR_P
         8 OBSERVE_PHA 9 OBSERVE_NOT   (act = rung number; lowest set bit wins; rung 9 = FLOOR)

   BUILD: cc -O2 -fwrapv. int32 arithmetic wraps at every step, exactly as the synthesis evaluator wraps.
*/
#include <stdint.h>
#define LANE static inline __attribute__((always_inline)) int32_t

LANE I_HI      (int32_t x){ return ((x >> 31) | ((-x) >> 31)); }
LANE INV_EQ1   (int32_t x){ return ((-x) >> 31); }
LANE CERT_GE5  (int32_t x){ return ((-(x - 16)) >> 31); }
LANE PB_EQ3    (int32_t x){ return (((-384) >> 31) ^ ((-(x ^ 384)) >> 31)); }
LANE FAINT_EQ1 (int32_t x){ return ((-x) >> 31); }
LANE CERT_EQ0  (int32_t x){ return ((x - 1) >> 31); }
LANE UNC_EQ0   (int32_t x){ return ((x - 1) >> 31); }
LANE UNC_EQ2   (int32_t x){ return (((-64) >> 31) ^ ((-(x ^ 64)) >> 31)); }
LANE CERT_IN24 (int32_t x){ return (((-(x & 12)) >> 31) ^ ((-(x & (x ^ 8))) >> 31)); }
LANE PB_EQ0    (int32_t x){ return ((x - 1) >> 31); }
LANE CERT_IN13 (int32_t x){ return (((-(x ^ 4)) >> 31) ^ ((-(x ^ 12)) >> 31)); }
LANE PB_EQ2    (int32_t x){ return (((-256) >> 31) ^ ((-(x ^ 256)) >> 31)); }
LANE CERT_IN12 (int32_t x){ return (((-(x ^ 4)) >> 31) ^ ((-(x ^ 8)) >> 31)); }
LANE CERT_EQ4  (int32_t x){ return (((-16) >> 31) ^ ((-(x ^ 16)) >> 31)); }
LANE AND2      (int32_t x){ return ((-(x - 2)) >> 31); }

int32_t pha_law(int32_t n)
{   int32_t inv = n & 0x1, faint = n & 0x2, cert = n & 0x1C, unc = n & 0x60, pb = n & 0x180, hi = n & ~0x1FF;
    int32_t m = ((I_HI(hi) | INV_EQ1(inv) | CERT_GE5(cert) | PB_EQ3(pb)) & 1)
              | ((FAINT_EQ1(faint)) & 2)
              | ((CERT_EQ0(cert)) & 4)
              | ((UNC_EQ0(unc)) & 8)
              | ((UNC_EQ2(unc)) & 16)
              | ((AND2((CERT_IN24(cert) & 1) | (PB_EQ0(pb) & 2))) & 32)
              | ((AND2((CERT_IN13(cert) & 1) | (PB_EQ2(pb) & 2))) & 64)
              | ((CERT_IN12(cert)) & 128)
              | ((CERT_EQ4(cert)) & 256)
              | 512;
    return __builtin_ctz((uint32_t)m);
}

/* ---- self-check (cc -O2 -fwrapv -DPHA_LAW_MAIN): the law against an independent C transcription of the
        specification (spec.py decide), over every one of the 2^32 int32 inputs. */
#ifdef PHA_LAW_MAIN
#include <stdio.h>
static int32_t spec_c(int32_t n)
{   if (n < 0 || (n >> 9)) return 0;
    int inv = n & 1, faint = (n >> 1) & 1, cert = (n >> 2) & 7, unc = (n >> 5) & 3, pb = (n >> 7) & 3;
    if (inv || cert >= 5 || pb == 3) return 0;
    if (faint) return 1;
    if (cert == 0) return 2;
    if (unc == 0) return 3;
    if (unc == 2) return 4;
    if ((cert == 2 || cert == 4) && pb == 0) return 5;
    if ((cert == 1 || cert == 3) && pb == 2) return 6;
    if (cert == 1 || cert == 2) return 7;
    if (cert == 4) return 8;
    return 9;
}
int main(void)
{   uint64_t bad = 0, cnt[10] = {0}; uint32_t u = 0;
    do { int32_t n = (int32_t)u, a = pha_law(n);
         if (a != spec_c(n)) { if (bad < 5) printf("MISMATCH n=%d law=%d spec=%d\n", n, a, spec_c(n)); bad++; }
         if (a >= 0 && a < 10) cnt[a]++;
    } while (++u != 0);
    printf("pha_law vs specification over all 4294967296 int32 inputs: %llu mismatches\n", (unsigned long long)bad);
    for (int k = 0; k < 10; k++) printf("  act %d: %llu\n", k, (unsigned long long)cnt[k]);
    return bad != 0;
}
#endif
