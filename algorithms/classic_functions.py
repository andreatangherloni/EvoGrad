import torch

# global device 

# device = "cpu" 

# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda" 
# else:
#     device = device

def Ackley(x, a=20, b=0.2, c=2 * torch.pi):
    # x = x.to(device)

    d = x.size(1)
    sum_sq_term = torch.sum(x ** 2, dim=-1) / d
    cos_term = torch.sum(torch.cos(c * x), dim=-1) / d

    term1 = -a * torch.exp(-b * torch.sqrt(sum_sq_term))
    term2 = -torch.exp(cos_term)

    return term1 + term2 + a + torch.exp(torch.tensor(1.0, device=x.device))

def Mishra01(x):
    # x = x.to(device)
    d = x.size(1)
    x_n = d - torch.sum(x[:, :-1], dim=-1)
    base = 1 + x_n
    base = torch.clamp(base, min=1e-6, max=100.0)  
    output = base ** x_n

    return torch.clamp(output, max=1e6) 

def Quintic(x):
    # x = x.to(device)

    poly = x**5 + 3*x**4 + 4*x**3 + 2*x**2 - 10*x - 4
    out = torch.abs(poly)
    
    if x.dim() == 1:
        return torch.sum(out)  
    else:
        return torch.sum(out, dim=-1)  
    
def Michalewicz(x, M = 10.0):
    # x = x.to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0) 

    d = x.size(1)
    i_tensor = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)

    sin_term = torch.sin(x)
    power_term = torch.sin((i_tensor * x**2) / torch.pi) ** (2.0 * M)

    fitness = -torch.sum(sin_term * power_term, dim=-1)
    return fitness
    # return fitness if fitness.shape[0] > 1 else fitness[0]

def Schubert(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    j_tensor = torch.arange(1, 6, dtype=x.dtype, device=x.device)  

    particles_exp = x.unsqueeze(2)  
    j_tensor = j_tensor.view(1, 1, -1)       
    expr = j_tensor * torch.cos((j_tensor + 1) * particles_exp + j_tensor)  
    summed = expr.sum(dim=2)                
    prod = torch.prod(summed, dim=-1)        
    return prod

def Alpine(x):

    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return torch.sum(torch.abs(x * torch.sin(x) + 0.1 * x), dim=-1)

def Bohachevsky(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    terms = x[ :, :-1]**2 + 2*x[:,1:]**2 - 0.3*torch.cos(3*torch.pi*x[:,:-1]) - 0.4*torch.cos(4*torch.pi*x[:,1:]) + 0.7
    return torch.sum(terms, dim=-1)

def Plateau(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return 30.0 + torch.sum(torch.floor(x), dim=-1)

def XinSheYang(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    numer = torch.sum(torch.abs(x), dim=-1)
    expo = torch.sum(torch.sin(x**2), dim=-1)
    denom = torch.exp(expo)
    return numer / denom

def Vincent(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    epsilon = 1e-10
    x = torch.clamp(x, min=epsilon)

    d = x.size(1)     
    return (1.0 / d) * torch.sum(torch.sin(10 * torch.log(x)), dim=-1)

def Vincent2(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = torch.clamp(x, min=0.25, max=10.0)
    return torch.sum(torch.sin(10 * torch.log(x)), dim=-1 if x.dim() == 2 else 0)

def Griewank(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    d = x.size(1)
    i_tensor = torch.arange(1, d + 1, dtype=x.dtype, device=x.device).view(1, -1)
    sum_term = torch.sum(x**2 / 4000, dim=-1)
    prod_term = torch.prod(torch.cos(x / torch.sqrt(i_tensor)), dim=-1)
    return 1.0 + sum_term - prod_term

def Rastrigin(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    A = 10.0
    d = x.size(1)
    
    return A * d + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x), dim=-1)

def Schwefel(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    d = x.size(1)
    return 418.9829 * d - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=-1)

def Rosenbrock(x):
    # x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    terms = 100.0 * (x[:, :-1]**2 - x[:, 1:])**2 + (x[:, :-1] - 1)**2
    return torch.sum(terms, dim=-1)