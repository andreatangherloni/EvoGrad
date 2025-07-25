import os, pickle, math, torch
from typing import Tuple

# ------------------------------------------------------------------
# 1. ── Load shift / rotation / shuffle data and convert to tensors ──
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(__file__)
with open(os.path.join(_THIS_DIR, "data.pkl"), "rb") as fh:
    _raw = pickle.load(fh)

_ROTS       = {d: torch.from_numpy(_raw[f"M_D{d}"]).float() for d in (2,10,20,30,50,100)}
_SHIFTS     = torch.from_numpy(_raw["shift"]).float()                    # (20, 100)
_SHUFFLES   = {d: torch.from_numpy(_raw[f"shuffle_D{d}"]).long()
                  for d in (10,30,50,100)}                               # for F11-F20

# ------------------------------------------------------------------
# 2. ── Helper: shift-rotate & shuffle-partition (batch-wise, torch) ─
# ------------------------------------------------------------------
def shift_rotate(x: torch.Tensor, shift: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """x: (B,D), shift: (D,), rot: (D,D) ➜ (B,D)"""
    return (x - shift) @ rot.T                                           # batch matmul

def shuffle_partition(x: torch.Tensor,
                      perm: torch.Tensor,
                      parts: Tuple[float, ...]) -> list:

    x = x[:, perm]                                   # (B, D)
    D = x.shape[1]

    # sizes for every block except the last
    sizes = [int(math.ceil(p * D)) for p in parts[:-1]]
    sizes.append(D - sum(sizes))                     # exact remainder

    segs, start = [], 0
    for sz in sizes:
        segs.append(x[:, start:start + sz])
        start += sz
    return segs

# ------------------------------------------------------------------
# 3. ── Basic building-block functions (torch, broadcast-able) ─────
# ------------------------------------------------------------------
def _bent_cigar(z):                       # (B,D)
    return z[...,0]**2 + 1e6*torch.sum(z[...,1:]**2, dim=-1)

def _sum_diff_pow(z):
    D = z.shape[-1]
    exps = 2 + 4*torch.arange(D, dtype=z.dtype, device=z.device)/(D-1)
    return torch.sum(torch.abs(z)**exps, dim=-1)

def _zakharov(z):
    i  = torch.arange(1, z.shape[-1]+1, dtype=z.dtype, device=z.device)
    term1 = torch.sum(z**2, dim=-1)
    term2 = torch.sum(0.5*i*z, dim=-1)
    return term1 + term2**2 + term2**4

def _rosenbrock(z):
    return torch.sum(100*(z[..., :-1]**2 - z[...,1:])**2 + (z[...,:-1]-1)**2, dim=-1)

def _rastrigin(z):
    return torch.sum(z**2 - 10*torch.cos(2*math.pi*z) + 10, dim=-1)

def _schaffers_f7(z):
    s   = z[..., :-1]**2 + z[..., 1:]**2
    t   = torch.sqrt(s) + 1e-16
    return torch.mean(torch.sqrt(t) * (torch.sin(50*t**0.2)**2 + 1), dim=-1)

def _lunacek_bi_rastrigin(z, m=5.0, d=1.0):
    D  = z.shape[-1]
    mu1 = 2.5
    mu2 = -math.sqrt((mu1**2 - d)/d)
    s1  = torch.sum((z - mu1)**2, dim=-1)
    s2  = torch.sum((z - mu2)**2, dim=-1)
    return torch.minimum(s1, d*D + s2) + 10*(D - torch.sum(torch.cos(2*math.pi*(z - mu1)), dim=-1))

def _non_cont_rastrigin(z):
    zc          = torch.where(torch.abs(z) > 0.5, torch.round(2*z)/2, z)
    return _rastrigin(zc)

def _levy(z):
    w  = 1 + (z - 1)/4
    s1 = torch.sin(math.pi*w[...,0])**2
    s2 = torch.sum((w[..., :-1]-1)**2 * (1+10*torch.sin(math.pi*w[..., :-1]+1)**2), dim=-1)
    s3 = (w[...,-1]-1)**2 * (1+torch.sin(2*math.pi*w[...,-1])**2)
    return s1 + s2 + s3

def _mod_schwefel(z):
    D = z.shape[-1]
    z = z + 4.209687462275036e+002  # classic 2017 offset
    out = 418.9829*D - torch.sum(z*torch.sin(torch.sqrt(torch.abs(z))), dim=-1)
    mask = (torch.abs(z) > 500)
    if mask.any():
        z_hat    = z.clone()
        z_hat[mask] = torch.remainder(z_hat[mask], 500)
        penalty = torch.sum((z_hat[mask])**2/(10000*D), dim=-1)
        out  += penalty
    return out

# Elliptic, Discus, Ackley, Weierstrass, Griewank, Katsuura,
# Happy-Cat, HGBat, Exp-Griewank+Rosenbrock, Exp-SchafferF6
# (compact, fully-vectorised versions follow)

def _elliptic(z):
    D   = z.shape[-1]
    coe = torch.pow(1e6, torch.linspace(0,1,D,device=z.device))
    return torch.sum(coe*z**2, dim=-1)

def _discus(z):
    return 1e6*z[...,0]**2 + torch.sum(z[...,1:]**2, dim=-1)

def _ackley(z):
    D = z.shape[-1]
    s1 = torch.mean(z**2, dim=-1)
    s2 = torch.mean(torch.cos(2*math.pi*z), dim=-1)
    return -20*torch.exp(-0.2*torch.sqrt(s1)) - torch.exp(s2) + 20 + math.e

def _weierstrass(z, a=0.5, b=3.0, kmax=20):
    D = z.shape[-1]
    k = torch.arange(0, kmax+1, device=z.device).view(1,1,-1)
    term1 = torch.sum(a**k * torch.cos(2*math.pi*b**k*(z.unsqueeze(-1)+0.5)), dim=-1)
    term2 = torch.sum(a**k * torch.cos(2*math.pi*b**k*0.5))
    return torch.sum(term1, dim=-1) - D*term2

def _griewank(z):
    part1 = torch.sum(z**2, dim=-1)/4000.0
    i     = torch.arange(1, z.shape[-1]+1, device=z.device)
    part2 = torch.prod(torch.cos(z/torch.sqrt(i)), dim=-1)
    return part1 - part2 + 1

def _katsuura(z):
    D   = z.shape[-1]
    j   = torch.arange(1,33, device=z.device).view(1,1,-1)
    tmp = torch.abs(2**j * z.unsqueeze(-1) - torch.round(2**j * z.unsqueeze(-1)))/(2**j)
    prod = torch.prod(1 + (j.float())*tmp, dim=-1)
    return (torch.sum(prod, dim=-1) - 1)*10/(D**1.2)

def _happy_cat(z, alpha=1/8):
    D   = z.shape[-1]
    norm = torch.sum(z**2, dim=-1)
    s1   = torch.pow(torch.abs(norm - D), 2*alpha) + (0.5*norm + torch.sum(z, dim=-1))/D
    return s1 + 0.5

def _hg_bat(z):
    D = z.shape[-1]
    norm = torch.sum(z**2, dim=-1)
    return torch.pow(torch.abs(norm**2 - torch.sum(z, dim=-1)**2), 0.5) + (0.5*norm + torch.sum(z, dim=-1))/D + 0.5

def _exp_rosen_griewank(z):
    # build zi = Rosenbrock(x_i, x_{i+1}), wrap with Griewank
    xi, xnext = z[...,:-1], z[...,1:]
    g = 100*(xi**2 - xnext)**2 + (xi - 1)**2
    return _griewank(g)  # note: g has D-1 dimensions

def _exp_schaffer_f6(z):
    x1, x2 = z[...,:-1], z[...,1:]
    f = 0.5 + (torch.sin(torch.sqrt(x1**2 + x2**2))**2 - 0.5) / \
        (1 + 0.001*(x1**2 + x2**2))**2
    return torch.sum(f, dim=-1)

# ------------------------------------------------------------------
# 4. ── Simple functions F1 … F10  (bias = i*100)  ──────────────────
# ------------------------------------------------------------------
def _simple_factory(idx, base_fn):
    def f(x: torch.Tensor, *, device=None):
        x = x.to(device or x.device)
        D = x.shape[1]
        rot   = _ROTS[D][idx].to(x.device)          # (D,D)
        shift = _SHIFTS[idx, :D].to(x.device)       # (D,)
        z     = shift_rotate(x, shift, rot)
        return base_fn(z) + 100.*(idx+1)
    return f

F1  = _simple_factory(0,  _bent_cigar)
F2  = _simple_factory(1,  _sum_diff_pow)
F3  = _simple_factory(2,  _zakharov)
F4  = _simple_factory(3,  _rosenbrock)
F5  = _simple_factory(4,  _rastrigin)
F6  = _simple_factory(5,  _schaffers_f7)
F7  = _simple_factory(6,  _lunacek_bi_rastrigin)
F8  = _simple_factory(7,  _non_cont_rastrigin)
F9  = _simple_factory(8,  _levy)
F10 = _simple_factory(9,  _mod_schwefel)

# ------------------------------------------------------------------
# 5. ── Hybrid functions F11 … F20  (bias = 1100 … 2000) ────────────
# ------------------------------------------------------------------
def _hybrid_factory(idx, blocks, bias):
    """blocks = [(weight, base_fn), ...] weights are split-fractions."""
    shuffle_idx = idx - 10                                               # 0-based for shuffles
    def f(x: torch.Tensor, *, device=None):
        x = x.to(device or x.device)
        D = x.shape[1]
        rot   = _ROTS[D][idx].to(x.device)
        shift = _SHIFTS[idx, :D].to(x.device)
        shuffle = _SHUFFLES[D][shuffle_idx].to(x.device)                 # (D,)
        z = shift_rotate(x, shift, rot)
        parts = shuffle_partition(z, shuffle, tuple(w for w,_ in blocks))
        out = torch.zeros(x.shape[0], device=x.device)
        for part, (_, fn) in zip(parts, blocks):
            out = out + fn(part)
        return out + bias
    return f

F11 = _hybrid_factory(10, [(0.2,_zakharov), (0.4,_rosenbrock), (0.4,_rastrigin)], 1100.)
F12 = _hybrid_factory(11, [(0.3,_elliptic), (0.3,_mod_schwefel), (0.4,_bent_cigar)], 1200.)
F13 = _hybrid_factory(12, [(0.3,_bent_cigar), (0.3,_rosenbrock), (0.4,_lunacek_bi_rastrigin)], 1300.)
F14 = _hybrid_factory(13, [(0.2,_elliptic), (0.2,_ackley), (0.2,_schaffers_f7), (0.4,_rastrigin)], 1400.)
F15 = _hybrid_factory(14, [(0.2,_bent_cigar), (0.2,_hg_bat), (0.3,_rastrigin), (0.3,_rosenbrock)], 1500.)
F16 = _hybrid_factory(15, [(0.2,_exp_schaffer_f6), (0.2,_hg_bat),
                           (0.3,_rosenbrock), (0.3,_mod_schwefel)], 1600.)
F17 = _hybrid_factory(16, [(0.1,_katsuura), (0.2,_ackley), (0.2,_exp_rosen_griewank),
                           (0.2,_mod_schwefel), (0.3,_rastrigin)], 1700.)
F18 = _hybrid_factory(17, [(0.2,_elliptic), (0.2,_ackley), (0.2,_rastrigin),
                           (0.2,_hg_bat), (0.2,_discus)], 1800.)
F19 = _hybrid_factory(18, [(0.2,_bent_cigar), (0.2,_rastrigin),
                           (0.2,_exp_rosen_griewank), (0.2,_weierstrass),
                           (0.2,_exp_schaffer_f6)], 1900.)
F20 = _hybrid_factory(19, [(0.1,_happy_cat), (0.1,_katsuura), (0.2,_ackley),
                           (0.2,_rastrigin), (0.2,_mod_schwefel),
                           (0.2,_schaffers_f7)], 2000.)