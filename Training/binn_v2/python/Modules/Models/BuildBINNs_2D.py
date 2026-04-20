"""Core 2D BINN model components and loss functions.

This module defines:
- input samplers for PDE residual points,
- data and PDE losses used during training,
- MLP heads for state (`u`), diffusion (`D`), and growth (`G`),
- and the `BINN_2d` model that combines these pieces.

The implementation below is intentionally direct to make the training equations
traceable from code to model assumptions.
"""

import sys, torch, pdb

sys.path.append('../../../../') # if inside binn/python/pipeline/exc
sys.path.append('../../../')
sys.path.append('../../')
import numpy as np
import torch.nn as nn

# From training
from binn_v2.python.Modules.Models.BuildMLP import BuildMLP
from binn_v2.python.Modules.Utils.Gradient import Gradient

    
# ============================================================
# Input Generation (2D)
# ============================================================

def generate_random_inputs_2d(self, inputs):
    """
    Generate random input samples for a 2D problem, scaled to specified ranges.

    Returns:
        torch.Tensor: (num_samples, 3) with columns [x1, x2, t]
    """
    torch.manual_seed(self.loss_count)
    torch.cuda.manual_seed(self.loss_count)

    # draw in [0,1]^3 then scale
    unit = torch.rand(self.num_samples, 3, requires_grad=True, device=inputs.device)

    x1 = unit[:, 0:1] * (self.x1_max - self.x1_min) + self.x1_min
    x2 = unit[:, 1:2] * (self.x2_max - self.x2_min) + self.x2_min
    t  = unit[:, 2:3] * (self.t_max  - self.t_min)  + self.t_min

    return torch.cat([x1, x2, t], dim=1).float()


def generate_lhs_inputs_2d(self, inputs, centered: bool = False):
    """
    Latin Hypercube for 2D space + time (x1, x2, t) using PyTorch only.

    Returns:
        torch.Tensor: (num_samples, 3) with columns [x1, x2, t]
    """
    N, D = self.num_samples, 3
    gen = torch.Generator(device="cpu").manual_seed(int(self.loss_count))

    unit = torch.empty(N, D, dtype=torch.float32)
    for j in range(D):
        perm = torch.randperm(N, generator=gen, dtype=torch.int64)
        if centered:
            coord = (perm.to(torch.float32) + 0.5) / N
        else:
            coord = (perm.to(torch.float32) + torch.rand(N, generator=gen)) / N
        unit[:, j] = coord

    unit = unit.to(inputs.device).requires_grad_(True)

    x1 = unit[:, 0:1] * (self.x1_max - self.x1_min) + self.x1_min
    x2 = unit[:, 1:2] * (self.x2_max - self.x2_min) + self.x2_min
    t  = unit[:, 2:3] * (self.t_max  - self.t_min)  + self.t_min

    return torch.cat([x1, x2, t], dim=1).float()


def generate_sobol_inputs_2d(self, inputs):
    """
    Sobol sampling (PyTorch) for 2D space + time.
    """
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=self.loss_count)
    unit = engine.draw(self.num_samples).to(inputs.device).requires_grad_(True)  # (N,3) in [0,1]

    x1 = unit[:, 0:1] * (self.x1_max - self.x1_min) + self.x1_min
    x2 = unit[:, 1:2] * (self.x2_max - self.x2_min) + self.x2_min
    t  = unit[:, 2:3] * (self.t_max  - self.t_min)  + self.t_min

    return torch.cat([x1, x2, t], dim=1).float()


# ============================================================
# Loss Application Helper
# ============================================================
def apply_constraints(self, D, G, t, u):
    """
    Apply biological constraints on diffusion, growth, and delay terms to penalize invalid values.

    Args:
        self: The object instance containing constraint configurations.
        D (torch.Tensor): Diffusion values.
        G (torch.Tensor): Growth values.
        t (torch.Tensor): Temporal input values.
        u (torch.Tensor): Output values from the model.

    Modifies:
        self.D_loss, self.G_loss, self.T_loss: Updated penalty terms.
    """
    self.D_loss = 0
    self.G_loss = 0

    if not self.allConstraints:
        return

    #Diffusion constraints
    self.D_loss += self.D_weight*torch.where(
           D <  self.alpha_D_min, (D -self.alpha_D_min)**2, torch.zeros_like(D))

    self.D_loss += self.D_weight*torch.where(
            D >  self.alpha_D_max, (D - self.alpha_D_max)**2, torch.zeros_like(D))
    
    if self.D_constraint_bool:
        try:
            dDdu = Gradient(D, u, order=1)
            self.D_loss += self.dDdu_weight*torch.where(
                    dDdu < 0.0, dDdu**2, torch.zeros_like(dDdu))
        except:
            pass
    
    if G is not None:
        # Growth constraints 
        self.G_loss += self.G_weight*torch.where(
                G < self.alpha_G_min, (G-self.alpha_G_min)**2, torch.zeros_like(G)) 
        self.G_loss += self.G_weight*torch.where(
                G > self.alpha_G_max, (G-self.alpha_G_max)**2, torch.zeros_like(G))

    try:
        dGdu = Gradient(G, u, order=1)

        self.G_loss += self.dGdu_weight*torch.where(
            dGdu > 0.0, dGdu**2, torch.zeros_like(dGdu))
    except:
             pass


# ============================================================
# No-flux BC input sampler (2D rectangle)
# ============================================================
def generate_bc_inputs_2d(self, inputs):
    """
    Draw `num_bcs` time stamps on [t_min,t_max] and place points on the four faces:
    x1 == x1_min or x1 == x1_max or x2 == x2_min or x2 == x2_max.

    Returns:
        torch.Tensor: (num_bcs, 3) with columns [x1, x2, t]
    """
    torch.manual_seed(self.loss_count)
    torch.cuda.manual_seed(self.loss_count)

    # times
    t = torch.rand(self.num_bcs, 1, requires_grad=True, device=inputs.device) \
        * (self.t_max - self.t_min) + self.t_min

    # which face: 0->x1_min, 1->x1_max, 2->x2_min, 3->x2_max
    face = torch.randint(0, 4, (self.num_bcs, 1), device=inputs.device)

    # free coordinates on each face
    x1_free = torch.rand(self.num_bcs, 1, device=inputs.device) * (self.x1_max - self.x1_min) + self.x1_min
    x2_free = torch.rand(self.num_bcs, 1, device=inputs.device) * (self.x2_max - self.x2_min) + self.x2_min

    # fixed coordinates per face
    x1_min = torch.full_like(t, self.x1_min)
    x1_max = torch.full_like(t, self.x1_max)
    x2_min = torch.full_like(t, self.x2_min)
    x2_max = torch.full_like(t, self.x2_max)

    # assemble (x1, x2) by face
    x1 = torch.where(face == 0, x1_min,
         torch.where(face == 1, x1_max,
         x1_free))
    x2 = torch.where(face == 2, x2_min,
         torch.where(face == 3, x2_max,
         x2_free))

    return torch.cat([x1, x2, t], dim=1)



def apply_BC_2d(self, inputs):
    """Sample boundary points and accumulate no-flux boundary-condition loss."""

    self.bc_loss = 0

    # apply BC
    inputs_bc = generate_bc_inputs_2d(self, inputs) 
    u_bc      = self.surface_fitter(inputs_bc)                 
    self.bc_loss += self.bc_weight * bc_no_flux_loss(self, inputs_bc, u_bc) 


# ============================================================
# PDE Loss (2D) – without BC
# ============================================================
def pde_loss_without_bc_2d(self, inputs, outputs):
    """
    inputs:  (N,3) -> [x1, x2, t]
    outputs: (N,1) -> u
    """
    x1 = inputs[:, 0:1]
    x2 = inputs[:, 1:2]
    t  = inputs[:, 2:3]

    u  = outputs.clone()
    d1 = Gradient(u, inputs, order=1)            # (N,3): [ux1, ux2, ut]
    ux1, ux2, ut = d1[:, 0:1], d1[:, 1:2], d1[:, 2:3]

    
    # D(u) or D(u,t)
    D = self.diffusion(u) if self.diffusion.inputs == 1 else self.diffusion(u, t)

    # ∇·(D ∇u) = ∂/∂x1(D*ux1) + ∂/∂x2(D*ux2)
    div = Gradient(D * ux1, inputs)[:, 0:1] + Gradient(D * ux2, inputs)[:, 1:2]

    LHS = ut
    if self.growth:
        G = self.growth(u) if self.growth.inputs == 1 else self.growth(u, t)
        RHS = self.D_max * div + self.G_max * G * u
    else:
        RHS = self.D_max * div
        G = None

    pde = (LHS - RHS).pow(2)

    apply_constraints(self, D, G, t, u)
    return pde + self.D_loss + self.G_loss

     

# ============================================================
# No-flux BC loss (2D)
# ============================================================
def bc_no_flux_loss_2d(self, inputs_bc, u_bc):
    """
    Penalise normal flux F = D(u) * ∂u/∂n on all rectangle faces.
    inputs_bc : (N,3) – boundary points (x1,x2,t)
    u_bc      : (N,1) – network prediction at those points
    """
    grads = Gradient(u_bc, inputs_bc, order=1)  # shape (N,3): [∂u/∂x1, ∂u/∂x2, ∂u/∂t]
    du_dx1 = grads[:, 0:1]
    du_dx2 = grads[:, 1:1+1]

    # Identify faces (exact equality is fine because we set them explicitly)
    on_x1_min = (inputs_bc[:, 0:1] == self.x1_min)
    on_x1_max = (inputs_bc[:, 0:1] == self.x1_max)
    on_x2_min = (inputs_bc[:, 1:2] == self.x2_min)
    on_x2_max = (inputs_bc[:, 1:2] == self.x2_max)

    # Normal derivative: x1-faces -> du/dx1; x2-faces -> du/dx2
    dudn = torch.zeros_like(du_dx1)
    dudn = torch.where(on_x1_min | on_x1_max, du_dx1, dudn)
    dudn = torch.where(on_x2_min | on_x2_max, du_dx2, dudn)

    # If you want physical flux with D(u): uncomment next two lines:
    # D_bc = self.diffusion(u_bc) if self.diffusion.inputs == 1 else self.diffusion(u_bc, inputs_bc[:, 2:3])
    # flux = self.D_max * D_bc * dudn;  return self.bc_weight * flux.pow(2)

    return self.bc_weight * dudn.pow(2)

def pde_loss_with_bc_2d(self, inputs, outputs):
    """PDE residual with additional no-flux boundary-condition penalty."""
    self.pde_loss_val = 0
    self.bc_loss_val_total = 0

    x1 = inputs[:, 0:1]
    x2 = inputs[:, 1:2]
    t  = inputs[:, 2:3]

    u  = outputs.clone()
    d1 = Gradient(u, inputs, order=1)
    ux1, ux2, ut = d1[:, 0:1], d1[:, 1:2], d1[:, 2:3]

    D = self.diffusion(u) if self.diffusion.inputs == 1 else self.diffusion(u, t)
    div = Gradient(D * ux1, inputs)[:, 0:1] + Gradient(D * ux2, inputs)[:, 1:2]

    LHS = ut
    if self.growth:
        G = self.growth(u) if self.growth.inputs == 1 else self.growth(u, t)
        RHS = self.D_max * div + self.G_max * G * u
    else:
        RHS = self.D_max * div
        G = None

    pde = (LHS - RHS).pow(2)

    apply_constraints(self, D, G, t, u)
    apply_BC(self, inputs)

    return pde + self.D_loss + self.G_loss + self.bc_loss


# def _ic_weights(self):
#     """
#     Pointwise weights for enforcing the initial condition on the current batch.
#     Returns a (N,1) tensor w where w = 1 + IC_weight on t == t_min (within tol),
#     else w = 1.
#     """
#     # times from the batch that produced `pred`
#     t = self.inputs[:, 2:3]
#     # tolerate tiny numeric noise (you normalized t to [0,1], so ~1e-6 is fine)
#     tol = getattr(self, "ic_tol", 1e-6)
#     ic_mask = (t - self.t_min).abs() <= tol
#     w = torch.ones_like(t)
#     w = w + self.IC_weight * ic_mask.float()
#     return w  # shape (N,1)


def data_loss_MSE(self, pred, true):
    """Pointwise mean-squared error term used for supervised surface fitting."""
    base = (pred - true).pow(2)          # (N,1)
    #if hasattr(self, "inputs"):
     #   return self._ic_weights() * base # up-weight IC residuals
    return base


def data_loss_GLS(self, pred, true):
    """Generalized least-squares loss with heteroscedastic weighting."""
    residual = (pred - true).pow(2)                                 # (N,1)
    residual *= pred.abs().clamp(min=1e-10).pow(-2 * self.gamma)    # GLS
   # if hasattr(self, "inputs"):
       # return self._ic_weights() * residual                        # IC weight
    return residual



# ============================================================
# ============================================================
# MLPS
# ============================================================
# ============================================================


class u_MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for predicting the solution of a governing PDE.

    This model includes three hidden layers with 128 sigmoid-activated neurons.
    The output layer uses a softplus activation to ensure non-negative cell densities.

    Attributes:
        scale (float): Output scaling factor, defaults to carrying capacity.
        mlp (nn.Module): The MLP model constructed with the specified layers and activations.

    Args:
        input_features (int, optional): Number of input features, defaults to 3.
        scale (float, optional): Scaling factor for the output, defaults to 1.7e3.

    Methods:
        forward(inputs): Computes the scaled MLP output.

    Inputs:
        inputs (torch.Tensor): Input tensor of shape (N, input_features).

    Returns:
        torch.Tensor: Predicted values of shape (N, 1).
    """
    def __init__(self, input_features=2, u_size=64):
        super().__init__()
        self.size = u_size
        self.mlp = BuildMLP(
            input_features=input_features,
            layers=[self.size, self.size, self.size, 1],
            activation=nn.SiLU(),
            linear_output=False,
            output_activation=nn.Softplus(),
            seed=0
        )

    def forward(self, inputs):
        # add dimensionality using self.scale
        return self.mlp(inputs)     

class D_MLP(nn.Module):  
    """
    Multi-Layer Perceptron (MLP) for predicting an unknown diffusivity function.

    This model includes three hidden layers with 32 sigmoid-activated neurons.
    The output layer uses a softplus activation to ensure non-negative diffusivities.

    Attributes:
        inputs (int): Number of input features.
        scale (float): Input scaling factor.
        min (float): Minimum diffusivity value.
        max (float): Maximum diffusivity value.
        mlp (nn.Module): The MLP model constructed with the specified layers and activations.

    Args:
        input_features (int, optional): Number of input features, defaults to 1.
        scale (float, optional): Scaling factor for inputs, defaults to 1.7e3.

    Methods:
        forward(u, t=None): Computes the scaled MLP output for diffusivity.

    Inputs:
        u (torch.Tensor): Predicted `u` values of shape (N, 1).
        t (torch.Tensor, optional): Optional time values of shape (N, 1).

    Returns:
        torch.Tensor: Predicted diffusivities of shape (N, 1).
    """
    def __init__(self, input_features=1, scale=1.7e3, D_size=4,
                use_single_bias=False, D_bound=0.01):
        super().__init__()
        self.inputs = input_features
        self.min = 0 / (1000**2) / (1/24) # um^2/hr -> mm^2/d
        self.max = D_bound #4000 / (1000**2) / (1/24) # um^2/hr -> mm^2/d

        self.size =  D_size
        #print(f"Using D_MLP with size {self.size} and use_single_bias={use_single_bias}")
        #from inspect import signature

        self.mlp = BuildMLP(
            input_features=input_features, 
            layers=[self.size, self.size, self.size, 1],
            activation=nn.SiLU(), 
            linear_output=False,
            output_activation=nn.Softplus(),
            seed=1,
            use_single_bias=use_single_bias)

    def forward(self, u):
        return self.mlp(u)  # u/self.scale to non-dimensionalize inputs

class G_MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for predicting an unknown growth function.

    This model includes three hidden layers with 32 sigmoid-activated neurons.
    The output layer uses a linear activation to allow both positive and negative growth rates.

    Attributes:
        inputs (int): Number of input features.
        scale (float): Input scaling factor.
        min (float): Minimum growth rate value.
        max (float): Maximum growth rate value.
        mlp (nn.Module): The MLP model constructed with the specified layers and activations.

    Args:
        input_features (int, optional): Number of input features, defaults to 1.
        scale (float, optional): Scaling factor for inputs, defaults to 1.7e3.

    Methods:
        forward(u, t=None): Computes the scaled MLP output for growth rates.

    Inputs:
        u (torch.Tensor): Predicted `u` values of shape (N, 1).
        t (torch.Tensor, optional): Optional time values of shape (N, 1).

    Returns:
        torch.Tensor: Predicted growth rates of shape (N, 1).
    """
    def __init__(self, input_features=1, G_size= 4):
        super().__init__()
        self.inputs = input_features
        self.min =  0#.02 / (1/24)  # 1/hr -> 1/d
        self.max = 2#0.1 / (1/24)  # 1/hr -> 1/d
        
        self.size =  G_size
        self.mlp = BuildMLP(
            input_features=input_features,
            layers=[self.size, self.size, self.size, 1],
            activation=nn.SiLU(),
            linear_output=True,
            seed=2
        )
    def forward(self, u, t=None):
        return self.mlp(u)
    

# ============================================================
# ============================================================
# BINN
# ============================================================
# ============================================================

import torch
import torch.nn as nn

class BINN_2d(nn.Module):
    """
    Biologically-Informed Neural Network (BINN) for modeling cell density, diffusion, and growth.

    This class implements a neural network that combines cell density-dependent diffusion and growth MLPs 
    with an optional time delay MLP. It supports modeling and loss computation for biologically informed simulations.

    Attributes:
        surface_fitter (u_MLP): Multi-Layer Perceptron (MLP) for modeling cell density.
        diffusion (D_MLP): MLP for modeling diffusion.
        growth (G_MLP): MLP for modeling growth, included if growth=True.
        delay1 (T_MLP or NoDelay): MLP for modeling time delay, included if delay=True.
        delay2 (T_MLP or NoDelay): Duplicate of delay1 for convenience.
        D_min (float): Minimum diffusion value.
        D_max (float): Maximum diffusion value.
        G_min (float): Minimum growth value (if growth is enabled).
        G_max (float): Maximum growth value (if growth is enabled).
        K (float): Constant for proportionality in loss functions.
        x1_arr (ndarray): Array of x-coordinates for the domain.
        x2_arr (ndarray): Array of y-coordinates for the domain (optional for 2D).
        t_arr (ndarray): Array of time points for the domain.
        x1_min (float): Minimum x-coordinate value.
        x1_max (float): Maximum x-coordinate value.
        x2_min (float or None): Minimum y-coordinate value (if x2_arr is provided).
        x2_max (float or None): Maximum y-coordinate value (if x2_arr is provided).
        t_min (float): Minimum time value.
        t_max (float): Maximum time value.
        IC_weight (float): Weight for initial condition in loss computation.
        surface_weight (float): Weight for the surface fitting loss term.
        pde_weight (float): Weight for the PDE loss term.
        D_weight (float): Weight scaling factor for diffusion term in loss.
        G_weight (float): Weight scaling factor for growth term in loss (if growth is enabled).
        T_weight (float): Weight scaling factor for time delay term in loss.
        dDdu_weight (float): Weight scaling factor for diffusion gradient term.
        dGdu_weight (float): Weight scaling factor for growth gradient term (if growth is enabled).
        dTdt_weight (float): Weight scaling factor for time delay derivative term.
        gamma (float): Proportionality constant for GLS weighting.
        num_samples (int): Number of samples used for PDE loss computation.
        name (str): Name of the model configuration based on included components.
        num_features (int): Number of input features (2 for 1D, 3 for 2D).
        pde_loss_func (callable): Function for computing PDE loss.
        inputs_gen_func (callable): Function for generating input samples.
        epochs (int): Number of training epochs completed.
        val_batch_it (int): Validation batch iteration count.
        tr_batch_it (int): Training batch iteration count.
        loss_count (int): Total loss computations performed.

    Args:
        x1_arr (ndarray): Array of x-coordinates.
        x2_arr (ndarray): Array of y-coordinates (optional for 2D).
        t_arr (ndarray): Array of time points.
        growth (bool, optional): Whether to include a growth MLP. Defaults to True.
        delay (bool, optional): Whether to include a time delay MLP. Defaults to False.

    Methods:
        __init__(self, x1_arr, x2_arr, t_arr, growth=True, delay=False):
            Initializes the BINN model with specified configurations.

        forward(self, inputs):
            Computes surface predictions using the cell density MLP.

        gls_loss(self, pred, true):
            Computes the Generalized Least Squares (GLS) loss for predictions.

        loss(self, pred, true):
            Computes the total loss combining GLS and PDE losses.
    """

    def __init__(self, 
                 data_obj_params,
                 model_params,
                 data_loss_func,
                 pde_loss_func):
        
        # DATA
        # ----------------------------------------------------------------------------------------------
        RDEq_params_store = data_obj_params["RDEq_params_store"]
        additional_params = data_obj_params["additional_params"] 
        #add_noise_params = data_obj_params["add_noise_params"] 
        # ----------------------------------------------------------------------------------------------
        # In BINN.__init__(...)
        xmax = RDEq_params_store["xmax"]
        ymax = RDEq_params_store["ymax"]
        dx = RDEq_params_store["dx"]
        dy = RDEq_params_store["dy"]
        t_arr = RDEq_params_store["hoursRange"]
        t_arr = np.array(t_arr) / 24  # normalize to [0,1]

        u_red_max = additional_params["u_red_max"]  # carrying capacity
        u_green_max = additional_params["u_green_max"]  # carrying capacity

        nx = max(1, int(np.ceil(xmax / dx)))
        ny = max(1, int(np.ceil(ymax / dy)))
        x_edges = np.linspace(0.0, xmax, nx + 1)
        y_edges = np.linspace(0.0, ymax, ny + 1)
        x1_arr = (x_edges[:-1] + x_edges[1:]) * 0.5
        x2_arr = (y_edges[:-1] + y_edges[1:]) * 0.5

        x1_arr /= 1000  # normalize to (0,1)
        x2_arr /= 1000  # normalize to (0,1)

        K = RDEq_params_store["K"] 
        gamma = RDEq_params_store["gamma"]

        # BINN
        # ----------------------------------------------------------------------------------------------
        binn_model_params = model_params["binn_model_params"]
        binn_construction_params = binn_model_params["binn_construction_params"]
        # ----------------------------------------------------------------------------------------------


        binnUsize = binn_construction_params["binnUsize"] 
        binnDsize = binn_construction_params["binnDsize"] 
        binnGsize = binn_construction_params["binnGsize"] 
        DoneParamBool = binn_construction_params["DoneParamBool"]
        D_constraint_bool = binn_construction_params["D_constraint_bool"]
        D_bound = binn_construction_params["D_bound"]
        allConstraints = binn_construction_params["allConstraints"]
        


        device = binn_construction_params['binnDevice']
        # loss
        numPDEsamples  = binn_model_params["pde_loss_params"]["numPDEsamples"] 

        
        super().__init__()

        # Initialize surface fitter and components
        self.surface_fitter = u_MLP(input_features=3, u_size=binnUsize).to(device)

        self.diffusion = D_MLP(D_size=binnDsize, D_bound=D_bound) if not DoneParamBool else D_MLP(D_size=binnDsize, D_bound=D_bound, use_single_bias=True)
        self.diffusion = self.diffusion.to(device)
        self.D_constraint_bool = D_constraint_bool
        self.allConstraints = allConstraints

        self.growth = G_MLP(G_size=binnGsize).to(device) if binnGsize else None


        # Parameter extrema
        self.u_red_max = u_red_max
        self.u_green_max = u_green_max
        self.D_min = self.diffusion.min
        self.D_max = self.diffusion.max
        self.D_scale = self.D_max 
        self.alpha_D_min = self.D_min/self.D_scale
        self.alpha_D_max = self.D_max/self.D_scale
        if self.growth:
            self.G_min = self.growth.min
            self.G_max = self.growth.max
            self.G_scale = self.G_max 
            self.alpha_G_min = self.G_min/self.G_scale
            self.alpha_G_max = self.G_max/self.G_scale
        self.K =  K             

        self.x1_arr = x1_arr
        self.x2_arr = x2_arr
        self.t_arr  = t_arr

        
        self.x1_min, self.x1_max = float(np.min(x1_arr)), float(np.max(x1_arr))
        self.x2_min, self.x2_max = float(np.min(x2_arr)), float(np.max(x2_arr))
        self.t_min,  self.t_max  = float(np.min(t_arr)),  float(np.max(t_arr))

        # Loss weights
        self.IC_weight = 1e0
        self.surface_weight = 1e0
        self.pde_weight = 1e0
        self.D_weight = 1e3 / self.D_max
        self.dDdu_weight = self.D_weight * self.K
        self.gamma = gamma
        self.num_samples = numPDEsamples   # change this


        self.diffusion_samples = 20

        if self.growth:
            self.G_min = self.growth.min
            self.G_max = self.growth.max
            self.G_scale = self.G_max 
            self.G_weight = 1e3 / self.G_max
            self.dGdu_weight = self.G_weight * self.K

        self.pde_loss_func = pde_loss_func
        self.data_loss_func = data_loss_func
        self.inputs_gen_func = generate_random_inputs_2d
        
        self.bc_weight   = 1e0          # tune if needed
        self.num_bcs     = 100          # # boundary points drawn each loss call


        # Training tracking
        self.epochs = 0
        self.val_batch_it = 0
        self.tr_batch_it = 0
        self.loss_count = 0
        self.pde_losses_all = {}
        self.inputs_all = {}


        # diffusion tracking
        self.u_vals = np.linspace(0,1,self.diffusion_samples)
        self.u_vals_torch = torch.tensor(self.u_vals, device=device, dtype=torch.float32).reshape(-1,1)

   

    def forward(self, inputs):
        """
        Forward pass for the model.

        Inputs:
            inputs (torch.Tensor): Input tensor for surface prediction.

        Returns:
            torch.Tensor: Predicted cell densities.
        """
        self.inputs = inputs
        return self.surface_fitter(self.inputs)



    def loss(self, pred, true):
        """
        Computes the total loss combining GLS and PDE losses.

        Inputs:
            pred (torch.Tensor): Predicted values.
            true (torch.Tensor): True values.

        Returns:
            tuple: Total loss, GLS loss, and PDE loss values.
        """
        
        self.data_loss_val = 0
        self.pde_loss_val = 0
        self.data_loss_val_total = 0
        self.pde_loss_val_total = 0

        # Load cached inputs from forward pass
        inputs = self.inputs

        # Generate random input samples for PDE loss
        inputs_rand = self.inputs_gen_func(self, inputs)
        outputs_rand = self.surface_fitter(inputs_rand)

        self.data_loss_val_total = self.data_loss_func(self, pred, true)
        self.pde_loss_val_total += self.pde_loss_func(self, inputs_rand, outputs_rand)


        # Compute mean GLS loss
        self.data_loss_val = self.surface_weight * torch.mean(self.data_loss_val_total)
        
        # Compute mean PDE loss
        self.pde_loss_val += self.pde_weight * torch.mean(self.pde_loss_val_total)
       
        self.loss_count += 1

        return (self.data_loss_val + self.pde_loss_val,
                self.data_loss_val,
                self.pde_loss_val)
    



    def freeze_surface(self, freeze: bool = True):
        """Toggle training of the surface_fitter (u_MLP) only."""
        for p in self.surface_fitter.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.surface_fitter.eval()   # no BN/Dropout anyway, but harmless

    def dg_parameters(self):
        """Params for D and (optionally) G heads only."""
        params = list(self.diffusion.parameters())
        if self.growth is not None:
            params += list(self.growth.parameters())
        return params
