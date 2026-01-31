import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

class LorentzSolver:
    """
    A solver for the motion of a charged particle in arbitrary E and B fields.
    Units used: SI (m, kg, s, C, T, V/m)
    """
    def __init__(self, charge, mass):
        self.q = charge
        self.m = mass
        # Default fields are zero
        self.B_field_func = lambda r, t: np.zeros(3)
        self.E_field_func = lambda r, t: np.zeros(3)

    def set_B_field(self, func):
        """Sets the magnetic field function B(r, t). Returns vector [Bx, By, Bz]."""
        self.B_field_func = func

    def set_E_field(self, func):
        """Sets the electric field function E(r, t). Returns vector [Ex, Ey, Ez]."""
        self.E_field_func = func

    def _equations_of_motion(self, t, state):
        """
        Differential equations for the solver:
        dr/dt = v
        dv/dt = (q/m) * (E + v x B)
        """
        # Unpack state: [x, y, z, vx, vy, vz]
        r = state[:3]
        v = state[3:]

        # Get local field values
        B = self.B_field_func(r, t)
        E = self.E_field_func(r, t)

        # Lorentz force: F = q(E + v x B)
        # Acceleration: a = (q/m)(E + v x B)
        lorentz_force = self.q * (E + np.cross(v, B))
        a = lorentz_force / self.m

        # Return derivatives [vx, vy, vz, ax, ay, az]
        return np.concatenate((v, a))

    def solve(self, t_span, initial_state, t_eval=None):
        """
        Solves the trajectory.
        t_span: (t_start, t_end)
        initial_state: [x0, y0, z0, vx0, vy0, vz0]
        """
        return solve_ivp(
            self._equations_of_motion,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45', # Runge-Kutta 4(5) is efficient for smooth trajectories
            rtol=1e-6,     # Relative tolerance
            atol=1e-9      # Absolute tolerance
        )

# --- Configuration Constants (SI Units) ---
# Physical Constants
ELECTRON_CHARGE = -1.602e-19  # Coulombs
ELECTRON_MASS = 9.109e-31     # kg

# Simulation Parameters
# 0.5 Tesla is a strong field, achievable with high-grade Neodymium magnets.
B_FIELD_STRENGTH = 0.5  # Tesla
VELOCITY_MAGNITUDE = 2e6 # 2,000,000 m/s (approx 0.7% speed of light - non-relativistic)

# Time settings
# The cyclotron frequency w_c = |q|B/m.
# Period T = 2*pi*m / (|q|B).
cyclotron_freq = (abs(ELECTRON_CHARGE) * B_FIELD_STRENGTH) / ELECTRON_MASS
period = 2 * np.pi / cyclotron_freq

# We want to simulate a few periods (turns of the helix)
num_periods = 4
total_time = num_periods * period
t_eval = np.linspace(0, total_time, 1000) # 1000 time steps for smooth plotting

# --- Setup Solver ---
solver = LorentzSolver(ELECTRON_CHARGE, ELECTRON_MASS)

# Define a uniform Magnetic Field in +Z direction
# B(r, t) = [0, 0, B0]
def uniform_B_field(r, t):
    return np.array([0.0, 0.0, B_FIELD_STRENGTH])

solver.set_B_field(uniform_B_field)

# --- Define Experiments (Different Initial Conditions) ---
experiments = []

# Experiment 1: Velocity parallel to B (Straight up)
exp1 = {
    'label': '0 rad (Parallel)',
    'color': 'blue',
    'pos': [0, 0, 0], # Start at origin
    'theta': 0 # Angle from Z-axis
}
experiments.append(exp1)

# Experiment 2: 30 degrees (pi/6) from vertical
exp2 = {
    'label': 'π/6 rad',
    'color': 'orange',
    'pos': [0.0005, 0, 0], # Shift x by 0.5mm
    'theta': np.pi/6
}
experiments.append(exp2)

# Experiment 3: 45 degrees (pi/4) from vertical
exp3 = {
    'label': 'π/4 rad',
    'color': 'green',
    'pos': [0.0010, 0, 0], # Shift x by 1.0mm
    'theta': np.pi/4
}
experiments.append(exp3)

# Experiment 4: 60 degrees (pi/3) from vertical
exp4 = {
    'label': 'π/3 rad',
    'color': 'red',
    'pos': [0.0015, 0, 0], # Shift x by 1.5mm
    'theta': np.pi/3
}
experiments.append(exp4)

# --- Run Simulations ---
results = []
print(f"Simulating {len(experiments)} electron trajectories...")
print(f"B-Field: {B_FIELD_STRENGTH} T (Uniform Z)")
print(f"Velocity: {VELOCITY_MAGNITUDE:.2e} m/s")

for exp in experiments:
    theta = exp['theta']
    pos = np.array(exp['pos'])
    
    # Calculate velocity components based on angle from Z-axis
    # v_z = v * cos(theta)
    # v_perp = v * sin(theta) -> We put this in x-direction for simplicity
    v_z = VELOCITY_MAGNITUDE * np.cos(theta)
    v_x = VELOCITY_MAGNITUDE * np.sin(theta)
    v_y = 0.0
    
    initial_state = np.concatenate((pos, [v_x, v_y, v_z]))
    
    sol = solver.solve((0, total_time), initial_state, t_eval=t_eval)
    
    if sol.success:
        results.append({'sol': sol, 'config': exp})
    else:
        print(f"Solver failed for {exp['label']}")

# --- Visualization ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories
for res in results:
    sol = res['sol']
    conf = res['config']
    
    # Convert meters to millimeters for easier reading on plot
    x_mm = sol.y[0] * 1e3
    y_mm = sol.y[1] * 1e3
    z_mm = sol.y[2] * 1e3
    
    ax.plot(x_mm, y_mm, z_mm, label=conf['label'], color=conf['color'], linewidth=1.5)
    
    # Mark start point
    ax.scatter(x_mm[0], y_mm[0], z_mm[0], color=conf['color'], s=20)

# --- Equalize X and Y Axes ---
# Get current data limits from the trajectories
x_limits = ax.get_xlim()
y_limits = ax.get_ylim()
z_limits = ax.get_zlim()

# Calculate the span of each axis
x_range = x_limits[1] - x_limits[0]
y_range = y_limits[1] - y_limits[0]
# Find the maximum span to enforce a square footprint
xy_max_range = max(x_range, y_range)

# Calculate midpoints
x_mid = np.mean(x_limits)
y_mid = np.mean(y_limits)

# Set new limits so both axes have the same span (xy_max_range) centered on their data
ax.set_xlim(x_mid - xy_max_range / 2, x_mid + xy_max_range / 2)
ax.set_ylim(y_mid - xy_max_range / 2, y_mid + xy_max_range / 2)

# Visualize B-Field vectors (Sparse grid to avoid clutter)
# We place a few arrows to indicate field direction
# Re-fetch limits now that they are equalized
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
z_min, z_max = ax.get_zlim()

# Create a small grid of arrows
# We place them at the 'back' of the plot roughly
x_grid = np.linspace(x_min, x_max, 3)
z_grid = np.linspace(z_min, z_max, 3)
# Fix Y to one side so arrows don't overlap data too much
y_plane = y_max 

for x in x_grid:
    for z in z_grid:
        # Direction is +Z (0, 0, 1) length scaled for visibility
        length = (z_max - z_min) * 0.1
        ax.quiver(x, y_plane, z, 0, 0, length, color='gray', alpha=0.3, arrow_length_ratio=0.3)

# Add a text label for B-field
ax.text(x_min, y_max, z_max, r"$\mathbf{B} = B_0 \hat{k}$", color='gray', fontsize=12)

# Labels and Styling
ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
ax.set_zlabel('Z Position (mm)')
ax.set_title(f'Electron Trajectories in Uniform Magnetic Field ({B_FIELD_STRENGTH} T)')
ax.legend()

# Set box aspect to 1:1 for X:Y
# Z is scaled by its ratio to the XY range to preserve relative shape, 
# or you can set it to a fixed value (e.g. 1.5) to compress/stretch Z for better viewing.
z_range = z_limits[1] - z_limits[0]
ax.set_box_aspect((1, 1, z_range / xy_max_range))

plt.tight_layout()
plt.show()
