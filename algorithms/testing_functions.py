import math, os, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------------

_THIS_DIR = os.path.dirname(__file__)
with open(os.path.join(_THIS_DIR, "data.pkl"), "rb") as fh:
    _raw = pickle.load(fh)

_ROTS = {}
for d in (2, 10, 20, 30, 50, 100):
    arr = np.asarray(_raw[f"M_D{d}"], dtype=np.float64) 
    if arr.ndim != 3 or arr.shape[1:] != (d, d):
        raise ValueError(f"Unexpected shape for M_D{d}: {arr.shape}, expected (20,{d},{d})")
    _ROTS[d] = torch.from_numpy(arr).float() 

_SHIFTS = torch.from_numpy(_raw["shift"]).float()

_SHUFFLES = {
    d: torch.from_numpy(_raw[f"shuffle_D{d}"].astype("int64"))
    for d in (10, 30, 50, 100)
}

# ------------------------------------------------------------------

def _pairwise_cyclic(x: torch.Tensor):
    """return (x_i, x_{i+1}) with wrap-around on the last dim."""
    return x, torch.roll(x, shifts=-1, dims=-1)

def _pairwise_adjacent(x: torch.Tensor):
    """return (x_i, x_{i+1}) for i=1..D-1 (no wrap-around)."""
    return x[..., :-1], x[..., 1:]

def _get_rot(D: int, *, device, dtype, func_id: int):
    """Return a (D,D) rotation matrix corresponding to `func_id` (1..20)."""
    Rstack = _ROTS.get(D, None)  
    if Rstack is None:
        return None
    idx = func_id - 1
    if not (0 <= idx < Rstack.shape[0]):
        raise IndexError(f"func_id {func_id} out of range for rotations of D={D}")
    R = Rstack[idx]  # (D, D)
    return R.to(device=device, dtype=dtype)

def apply_shift_rot(f_basic, func_id: int, D: int = 2):
    """
    transform: z = (x - o) @ R^T  
    """
    def f(x: torch.Tensor) -> torch.Tensor:
        shift = _SHIFTS[func_id - 1, :D].to(device=x.device, dtype=x.dtype)  
        z = x - shift
        R = _get_rot(D, device=x.device, dtype=x.dtype, func_id=func_id)     
        if R is not None:
            z = torch.matmul(z, R.transpose(-1, -2))
        return f_basic(z)
    return f



# ------------------------------------------------------------------

def _zakharov(z):
    term1 = torch.sum(z**2, dim=-1)
    term2 = 0.5 * torch.sum(z, dim=-1)
    return term1 + term2**2 + term2**4

def _rosenbrock(z):
    return torch.sum(100*(z[..., :-1]**2 - z[..., 1:])**2 + (z[..., :-1] - 1)**2, dim=-1)

def _rastrigin(z):
    return torch.sum(z**2 - 10*torch.cos(2*math.pi*z) + 10, dim=-1)

def _levy(z):
    w  = 1 + (z - 1)/4
    s1 = torch.sin(math.pi*w[..., 0])**2
    s2 = torch.sum((w[..., :-1]-1)**2 * (1+10*torch.sin(math.pi*w[..., :-1]+1)**2), dim=-1)
    s3 = (w[..., -1]-1)**2 * (1+torch.sin(2*math.pi*w[..., -1])**2)
    return s1 + s2 + s3

def _bent_cigar(z):
    return z[..., 0]**2 + 1e6*torch.sum(z[..., 1:]**2, dim=-1)

def _hg_bat(z):
    D = z.shape[-1]
    norm = torch.sum(z**2, dim=-1)
    return torch.pow(torch.abs(norm**2 - torch.sum(z, dim=-1)**2), 0.5) + (0.5*norm + torch.sum(z, dim=-1))/D + 0.5

def _elliptic(z):
    D   = z.shape[-1]
    coe = torch.pow(1e6, torch.linspace(0, 1, D, device=z.device))
    return torch.sum(coe*z**2, dim=-1)

def _happy_cat(z, alpha=1/8):
    D   = z.shape[-1]
    norm = torch.sum(z**2, dim=-1)
    s1  = torch.pow(torch.abs(norm - D), 2*alpha) + (0.5*norm + torch.sum(z, dim=-1))/D
    return s1 + 0.5

def _ackley(z):
    s1 = torch.mean(z**2, dim=-1)
    s2 = torch.mean(torch.cos(2*math.pi*z), dim=-1)
    return -20*torch.exp(-0.2*torch.sqrt(s1)) - torch.exp(s2) + 20 + math.e

def _discus(z):
    return 1e6*z[..., 0]**2 + torch.sum(z[..., 1:]**2, dim=-1)

def _griewank(z):
    part1 = torch.sum(z**2, dim=-1)/4000.0
    i     = torch.arange(1, z.shape[-1]+1, device=z.device, dtype=z.dtype)
    part2 = torch.prod(torch.cos(z/torch.sqrt(i)), dim=-1)
    return part1 - part2 + 1

def _exp_schaffer_f6(z):
    x1, x2 = _pairwise_cyclic(z) 
    r2 = x1**2 + x2**2
    g  = 0.5 + (torch.sin(torch.sqrt(r2))**2 - 0.5) / (1 + 0.001*r2)**2
    return torch.sum(g, dim=-1)

def _exp_rosen_griewank(z):
    xi, xnext = _pairwise_cyclic(z)
    r = 100*(xi**2 - xnext)**2 + (xi - 1)**2
    return torch.sum(r**2/4000.0 - torch.cos(r) + 1.0, dim=-1)

def _mod_schwefel(z):
    D = z.shape[-1]
    zi = z + 4.209687462275036e+002  
    m1 = (torch.abs(zi) <= 500)
    m2 = (zi > 500)
    m3 = (zi < -500)

    out = torch.zeros_like(zi)

    out[m1] = zi[m1] * torch.sin(torch.sqrt(torch.abs(zi[m1])))

    if m2.any():
        mod2 = torch.remainder(zi[m2], 500)
        t    = 500 - mod2
        out[m2] = t * torch.sin(torch.sqrt(torch.abs(t))) - ((zi[m2] - 500.0)**2)/(10000.0*D)

    if m3.any():
        mod3 = torch.remainder(torch.abs(zi[m3]), 500)
        t    = mod3 - 500.0
        out[m3] = t * torch.sin(torch.sqrt(torch.abs(t))) - ((zi[m3] + 500.0)**2)/(10000.0*D)

    return 418.9829*D - torch.sum(out, dim=-1)

def _katsuura(z):
    D = z.shape[-1]
    i = torch.arange(1, D+1, device=z.device, dtype=z.dtype).view(*(1,)* (z.ndim-1), D)
    j = torch.arange(1, 33, device=z.device, dtype=z.dtype).view(*(1,)*z.ndim, 32)
    pow2j = torch.pow(torch.tensor(2.0, dtype=z.dtype, device=z.device), j)

    z_exp = z.unsqueeze(-1) 
    frac = torch.abs(pow2j * z_exp - torch.round(pow2j * z_exp)) / pow2j   
    inner_sum = torch.sum(frac, dim=-1)  

    base = 1.0 + i * inner_sum
    power = 10.0 / (D**1.2)
    prod_term = torch.prod(torch.pow(base, power), dim=-1)

    return (10.0 / (D**2)) * (prod_term - 1.0)

def _schaffers_f7(z):
    xi, xnext = _pairwise_adjacent(z)
    s = torch.sqrt(xi**2 + xnext**2)
    mean_term = torch.mean(torch.sqrt(s) * (torch.sin(50.0 * s**0.2) + 1.0), dim=-1)
    return mean_term**2

# ------------------------------------------------------------------

def plot_function_2d(f, name, bounds=(-5, 5), res=200, device='cpu', log_scale=False, save_path=None):
    """ plots contour + 3D surface """
    x = torch.linspace(bounds[0], bounds[1], res)
    y = torch.linspace(bounds[0], bounds[1], res)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack((X, Y), dim=-1).to(device)

    with torch.no_grad():
        Z = f(grid).cpu().numpy()

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    cs = ax1.contourf(X.numpy(), Y.numpy(), Z, 50, cmap='viridis')
    ax1.set_title(name)
    plt.colorbar(cs, ax=ax1)
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X.numpy(), Y.numpy(), Z, cmap='viridis', linewidth=0, antialiased=True)
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_zlabel("f(x)")
    ax2.set_title(name + " Surface")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("shifted_rotated_plots", exist_ok=True)

    BOUNDS = (-100.0, 100.0)
    RES    = 200

    # map each function to its CEC row index in _SHIFTS
    FUNC_IDS = {
        "Zakharov":                          1,
        "Rosenbrock":                        2,
        "Expanded Schaffer F6":              3,
        "Rastrigin":                         4,
        "Levy":                              5,
        "Bent Cigar":                        6,
        "HGBat":                             7,
        "High-Conditioned Elliptic":         8,
        "Katsuura":                          9,
        "HappyCat":                          10,
        "Expanded Rosenbrock + Griewank":    11,
        "Modified Schwefel":                 12,
        "Ackley":                            13,
        "Discus":                            14,
        "Griewank":                          15,
        "Schaffer F7":                       16,
    }

    base_funcs = [
        (_zakharov,              "Zakharov"),
        (_rosenbrock,            "Rosenbrock"),
        (_exp_schaffer_f6,       "Expanded Schaffer F6"),
        (_rastrigin,             "Rastrigin"),
        (_levy,                  "Levy"),
        (_bent_cigar,            "Bent Cigar"),
        (_hg_bat,                "HGBat"),
        (_elliptic,              "High-Conditioned Elliptic"),
        (_katsuura,              "Katsuura"),
        (_happy_cat,             "HappyCat"),
        (_exp_rosen_griewank,    "Expanded Rosenbrock + Griewank"),
        (_mod_schwefel,          "Modified Schwefel"),
        (_ackley,                "Ackley"),
        (_discus,                "Discus"),
        (_griewank,              "Griewank"),
        (_schaffers_f7,          "Schaffer F7"),
    ]

    D = 2  
    for f_basic, name in base_funcs:
        func_id = FUNC_IDS[name]
        # f_trans = apply_shift_rot(f_basic, func_id=func_id, D=D)

        fname = name.replace(" ", "_").replace("+", "plus").replace("/", "_") 
        path  = f"basic_functions_plots_{RES}/{fname}.png"
        try:
            print(f"Plotting {name} → {path}")
            plot_function_2d(
                f_basic,
                f"{name}",
                bounds=BOUNDS,
                res=RES,
                save_path=path
            )
        except Exception as e:
            print(f"[WARN] Skipped {name}: {e}")
