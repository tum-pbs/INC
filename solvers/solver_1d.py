import torch
import math

def weno_reconstruction(f, direction='left'):
    eps = 1e-6 # from original paper
    # Periodic shifts for 5-point stencil
    f_m2 = torch.roll(f, shifts=2, dims=-1)
    f_m1 = torch.roll(f, shifts=1, dims=-1)
    f_0  = f
    f_p1 = torch.roll(f, shifts=-1, dims=-1)
    f_p2 = torch.roll(f, shifts=-2, dims=-1)
    f_p3 = torch.roll(f, shifts=-3, dims=-1)
    
    if direction == 'left':
        # Left-biased stencils (original)
        q0 = (1/3)*f_m2 - (7/6)*f_m1 + (11/6)*f_0
        q1 = (-1/6)*f_m1 + (5/6)*f_0 + (1/3)*f_p1
        q2 = (1/3)*f_0 + (5/6)*f_p1 - (1/6)*f_p2
        beta0 = (13/12)*(f_m2 - 2*f_m1 + f_0)**2 + (1/4)*(f_m2 - 4*f_m1 + 3*f_0)**2
        beta1 = (13/12)*(f_m1 - 2*f_0 + f_p1)**2 + (1/4)*(f_m1 - f_p1)**2
        beta2 = (13/12)*(f_0 - 2*f_p1 + f_p2)**2 + (1/4)*(3*f_0 - 4*f_p1 + f_p2)**2
        gamma = [0.1, 0.6, 0.3]
        
    elif direction == 'right':
        # Right-biased stencils (using i+1, i+2, i+3)
        q0 = (11/6)*f_p1 - (7/6)*f_0 + (1/3)*f_m1  # stencil: f_{i+1}, f_i, f_{i-1}
        q1 = (1/3)*f_p2 + (5/6)*f_p1 - (1/6)*f_0    # stencil: f_{i+2}, f_{i+1}, f_i
        q2 = - (1/6)*f_p3 + (5/6)*f_p2 + (1/3)*f_p1   # stencil: f_{i+3}, f_{i+2}, f_{i+1}
        
        beta0 = (13/12)*(f_p1 - 2*f_0 + f_m1)**2 + (1/4)*(3*f_p1 - 4*f_0 + f_m1)**2
        beta1 = (13/12)*(f_p2 - 2*f_p1 + f_0)**2 + (1/4)*(f_p2 - f_0)**2
        beta2 = (13/12)*(f_p3 - 2*f_p2 + f_p1)**2 + (1/4)*(3*f_p3 - 4*f_p2 + f_p1)**2
        gamma = [0.1, 0.6, 0.3]
        
    else:
        raise ValueError("Direction must be 'left' or 'right'.")
    
    # Compute nonlinear weights and return reconstructed flux
    alpha = torch.stack([g / (eps + b)**2 for g, b in zip(gamma, [beta0, beta1, beta2])])
    alpha_sum = alpha.sum(dim=0)
    w = alpha / alpha_sum
    return w[0]*q0 + w[1]*q1 + w[2]*q2

class BurgersSolverTorch:
    def __init__(self, sim_params,LOG=None):
        self.param = sim_params
        self.resolution = self.param.resolution
        self.dt = self.param.dt
        self.x = self.param.x
        self.t = 0.0
        self.adaptive_CFL = sim_params.adaptive_CFL
        self.correction_s=None
        self.L = self.param.L
        self.nu = self.param.nu
        self.device = self.param.device
        self.dx = self.param.L / self.param.resolution
        self.LOG = LOG
        self.D2_T = self._create_diffusion_matrix().T
        # Use either finite-difference or WENO for the convection term.
        if self.param.convection_term == "FDM":
            self.D1_T = self._create_convection_matrix().T
            self.calc_nonlinear = self.calc_nonlinear_FDM
        elif self.param.convection_term == "WENO":
            self.calc_nonlinear = self.calc_nonlinear_weno
        else:
            raise ValueError("Convection scheme not recognized")
        
    def initial_conditions(self, x=None, t=None):
        """Compute initial condition (both u_0 and s) from parameters"""
        x = self.x if x is None else x
        t = self.t if t is None else t
        return self._compute_delta(x, t)

    def _compute_delta(self, x, t):
        x_ = x.reshape(1, -1)  # Shape: [1, resolution]
        delta = torch.zeros(self.param.batch_size, x.shape[0], device=self.device)
        t = torch.full((self.param.batch_size,1), t, device=self.device) if isinstance(t, (int, float)) else t
        if t.shape != (self.param.batch_size,1):
            # raise ValueError("nu tensor must have shape [batch_size,1]")
            raise ValueError("t tensor must have shape [batch_size,1]")
        # Loop over all J components (waves)
        for j in range(self.param.J):
            arg = (self.param.omega[..., j] * t +(2 * torch.pi * self.param.l[..., j] / self.param.L) * x_ +self.param.phi[..., j])
            delta += self.param.A[..., j] * torch.sin(arg)
        return delta
    
    def _create_convection_matrix(self):
        """2nd order central difference matrix for convection derivative"""
        D1 = torch.zeros(self.resolution, self.resolution, device=self.device)
        for i in range(self.resolution):
            D1[i, (i-1) % self.resolution] = -1/(2*self.dx)
            D1[i, (i+1) % self.resolution] = 1/(2*self.dx)
        return D1

    def _create_diffusion_matrix(self):
        """2nd order central difference matrix for diffusion derivative"""
        D2 = torch.zeros(self.resolution, self.resolution, device=self.device)
        for i in range(self.resolution):
            D2[i, (i-1) % self.resolution] = 1/self.dx**2
            D2[i, i] = -2/self.dx**2
            D2[i, (i+1) % self.resolution] = 1/self.dx**2
        return D2
    
    def _compute_source(self, u):
        """Determine the source term based on the equation type."""
        if self.param.equation_type in ["homogeneous", "viscosity"]:
            source = 0.0
        elif self.param.equation_type == "linear":
            source = self.param.beta * u
        elif self.param.equation_type == "learning":
            source = self._compute_delta(self.x, self.t)
        else:
            raise ValueError("Unknown equation type.")
        return source
    
    def _apply_corrections(self, source):
        """
        If there are correction terms for the source (s), apply them.
        Port for coupling with NN or the experiments with different error.
        """
        if self.correction_s is not None:
            source = source + self.correction_s
        return source
    
    def calc_nonlinear_FDM(self, u):
        # return -u * (self.D1 @ u[..., None])[..., 0]
        return -u * (u @ self.D1_T)
    
    def calc_nonlinear_weno(self, u):
        """Compute the nonlinear convection term using WENO5 with Lax-Friedrichs splitting."""
        flux = 0.5 * u**2
        # alpha = torch.max(torch.abs(u))
        alpha = torch.max(torch.abs(u), dim=1, keepdim=True)[0]  # shape: [batch_size, 1]
        flux_plus  = 0.5 * (flux + alpha * u)
        flux_minus = 0.5 * (flux - alpha * u)
        flux_plus_hat  = weno_reconstruction(flux_plus, direction='left')
        flux_minus_hat = weno_reconstruction(flux_minus, direction='right')
        # Backward differences for both fluxes (from original paper)
        dflux_plus = (flux_plus_hat - torch.roll(flux_plus_hat, shifts=1, dims=-1)) / self.dx
        dflux_minus = (flux_minus_hat - torch.roll(flux_minus_hat, shifts=1, dims=-1)) / self.dx
        nonlinear_term = -(dflux_plus + dflux_minus)
        return nonlinear_term  
    
    def calc_linear(self, u):
        # return self.nu * (self.D2 @ u[..., None])[..., 0]
        return self.nu * (u @ self.D2_T)
    
    def step_adaptive(self, u):
        """
        Advance the solution by a total time self.dt using adaptive substeps.
        At each substep, the time step is chosen such that:
            dt_local <= CFL * dx / max(|u|)
        """
        target_dt = self.dt
        max_subsetps = 1000
        substep = 0
        while target_dt > 0:
            max_u = torch.max(torch.abs(u))
            # Compute allowed dt based on the CFL condition.
            if max_u.item() > 0:
                dt_allowed = self.adaptive_CFL * self.dx / max_u.item()
            else:
                dt_allowed = target_dt  # If velocity is zero everywhere, no CFL restriction.
            dt_local = min(target_dt, dt_allowed)
            source = self._compute_source(u)
            u,source = self._apply_corrections(u, source)
            nonlinear = self.calc_nonlinear(u)
            linear = self.calc_linear(u)
            rhs = nonlinear + linear + source
            u = u + dt_local * rhs
            self.t += dt_local
            target_dt -= dt_local
            if int(math.ceil(target_dt / dt_local)) > max_subsetps:
                if self.LOG is not None:
                    self.LOG.warning(f"Substeps {int(math.ceil(target_dt / dt_local))} exceeded max substeps ({max_subsetps})")
                break
            substep += 1
        self.LOG.info(f"CFL: {self.adaptive_CFL}, using substeps: {substep}, time_step: {self.dt}, max_u: {max_u}") if self.LOG is not None else None
        return u

    def step(self, u):
        """Advance the solution by one time interval (of length self.dt) using either a fixed or adaptive method."""
        if self.adaptive_CFL is not None:
            return self.step_adaptive(u)
        else:
            # Fixed time step update
            source = self._compute_source(u)
            u,source = self._apply_corrections(u, source)
            nonlinear = self.calc_nonlinear(u)
            linear = self.calc_linear(u)
            rhs = nonlinear + linear + source
            u_new = u + self.dt * rhs
            self.t += self.dt
            self.LOG.info(f"CFL = {torch.max(torch.abs(u)) * self.dt / self.dx:.2f}, time_step: {self.dt}") if self.LOG is not None else None
            return u_new


class KSSolverTorch:
    def __init__(self,sim_params):
        self.resolution = sim_params.resolution
        self.dt = sim_params.dt
        self.doublestep = True
        self.device = sim_params.device
        self.domain_size = sim_params.domain_size
        self.alpha = sim_params.alpha
        self.correction_s = None
        self.correction_u = None
        self.fft_indices = torch.fft.fftfreq(self.resolution, 
            d=1.0,  # Normalized spacing
            device=self.device) * self.resolution
        time_scheme_map = {'ETRK2':self.etrk2,'ETD1':self.etd1}
        self.step = time_scheme_map[sim_params.time_scheme]
        
    def initial_conditions(self):
        domain_size = self.domain_size.unsqueeze(1) # [B,1]
        alpha = self.alpha.unsqueeze(1) # [B,1]
        if len(alpha) != len(domain_size):
            raise ValueError("Length of alpha and domain_size must be the same")
        x =torch.linspace(0, 1, self.resolution, device=self.device)
        u = torch.cos(2 * torch.pi * x) + alpha * torch.sin(2 * torch.pi * x )
        epsilon = torch.randn_like(u)
        u = u + epsilon / torch.linalg.norm(epsilon)
        return u
    
    def calc_nonlinear(self, u, wavenumbers):
        """Computes nonlinear term in Fourier space"""
        # dealiasing
        N = u.shape[-1]
        M = 3 * N // 2 
        u_padded = torch.nn.functional.pad(u, (0, M - N), mode='constant', value=0)
        u_sq_padded = u_padded**2
        u_sq = u_sq_padded[..., :N]
        return -0.5 * wavenumbers * torch.fft.fft(u_sq)
    
    def calc_linear(self, domain_size):
        domain_size = domain_size.unsqueeze(-1)  # (B, 1)
        wavenumbers = (2j * torch.pi * self.fft_indices) / domain_size  # [B, resolution]
        L = -wavenumbers**2 - wavenumbers**4
        # Linear operator L = -(ik)^2 - (ik)^4 (i is included in k, so the sign should be negative)
        return L, wavenumbers

    def etd1_step(self, u, domain_size):
        L, wavenumbers = self.calc_linear(domain_size)
        # ETD1 coefficients
        exp_Ldt = torch.exp(L * self.dt)
        phi1 = torch.where(L == 0, self.dt * torch.ones_like(L),(exp_Ldt - 1) / L)
        
        # Step in Fourier space
        u_hat = torch.fft.fft(u)
        N_hat = self.calc_nonlinear(u, wavenumbers)
        e_s_hat = torch.fft.fft(self.correction_s) if self.correction_s is not None else 0.0

        u_new_hat = exp_Ldt * u_hat + phi1 * (N_hat + e_s_hat)
        # Return to physical space
        return torch.fft.ifft(u_new_hat).real
    
    def etrk2_step(self, u, domain_size):
        L, wavenumbers = self.calc_linear(domain_size)
        exp_Ldt = torch.exp(L * self.dt)
        
        phi1 = torch.where(torch.abs(L)<1e-6,
                          self.dt * torch.ones_like(L),
                          (exp_Ldt - 1) / L)

        phi2 = torch.where(torch.abs(L)<1e-6,
                          (self.dt/2) * torch.ones_like(L),
                          (exp_Ldt - 1 - L*self.dt) / (L**2 * self.dt))

        u_hat = torch.fft.fft(u)
        N1_hat = self.calc_nonlinear(u,wavenumbers)
        e_s_hat = torch.fft.fft(self.correction_s) if self.correction_s is not None else 0.0
        u_interm_hat = exp_Ldt * u_hat + phi1 * (N1_hat + e_s_hat)
        
        u_interm = torch.fft.ifft(u_interm_hat).real
        N2_hat = self.calc_nonlinear(u_interm,wavenumbers)
        u_new_hat = u_interm_hat + phi2 * (N2_hat - N1_hat)
        
        return torch.fft.ifft(u_new_hat).real

    def etd1(self, u, domain_size):
        u = u + self.correction_u if self.correction_u is not None else u
        u_next = self.etd1_step(u, domain_size)
        return self.etd1_step(u_next, domain_size) if self.doublestep else u_next

    def etrk2(self, u, domain_size):
        u = u + self.correction_u if self.correction_u is not None else u
        u_next = self.etrk2_step(u,domain_size)
        return self.etrk2_step(u_next,domain_size) if self.doublestep else u_next