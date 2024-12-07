import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import sys
import time
import os
from numba import njit, prange

# -----------------------------------------------------------------------------------------
# Physically realistic simulation of a 37,000 km space elevator collapse on Mars.
# Starting from stable configuration (cable along x-axis, rotating with Mars),
# at t=0 we "cut" the top anchor.
# -----------------------------------------------------------------------------------------

GM = 4.282837e13
R_mars = 3389500.0
J2 = 0.00196045
omega_mars = 7.088218e-05
M_mars = GM / 6.67430e-11

L = 3.7e7
rho_cable = 1300.0
A = 1e-4
E = 3.5e11
Cd = 1.0

rho0_atm = 0.02
H_atm = 11100.0

N = 1000
segment_length = L/N
m = rho_cable*A*segment_length
damping_alpha = 1e-4

dt = 0.1
Tfinal = 3600.0
num_steps = int(Tfinal/dt)

print("Using parallel numba if possible.")
print(f"N={N}, segment_length={segment_length} m, mass per node={m} kg")
print(f"dt={dt} s, total_steps={num_steps}")
print("Starting from stable elevator configuration (straight line, rotating with planet).")
print("Then cut top at t=0, watch collapse.")
sys.stdout.flush()

@njit(fastmath=True)
def gravity_j2(pos0,pos1,pos2):
    r = math.sqrt(pos0*pos0+pos1*pos1+pos2*pos2+1e-30)
    z = pos2
    phi = math.asin(z/r)
    g_factor = 1 - J2*(3*math.cos(phi)**2 -1)/2
    g_mag = GM/(r**2)*g_factor
    invr = 1.0/r
    gx = -(g_mag*invr)*pos0
    gy = -(g_mag*invr)*pos1
    gz = -(g_mag*invr)*pos2
    return gx,gy,gz

@njit(fastmath=True)
def atmosphere_density(pos0,pos1,pos2):
    r = math.sqrt(pos0*pos0+pos1*pos1+pos2*pos2+1e-30)
    alt = r - R_mars
    if alt<0:
        return rho0_atm
    return rho0_atm*math.exp(-alt/H_atm)

@njit(parallel=True, fastmath=True)
def compute_forces(X,V,N,m,E,A,segment_length,Cd,damping_alpha,omega_mars):
    F = np.zeros((N+1,3), dtype=np.float64)
    # Tension
    for i in prange(N):
        dx0 = X[i+1,0]-X[i,0]
        dx1 = X[i+1,1]-X[i,1]
        dx2 = X[i+1,2]-X[i,2]
        dist = math.sqrt(dx0*dx0+dx1*dx1+dx2*dx2+1e-30)
        strain = dist/segment_length - 1.0
        T = E*A*strain
        invd = 1.0/dist
        t0 = dx0*invd
        t1 = dx1*invd
        t2 = dx2*invd
        F[i,0]   += T*t0
        F[i,1]   += T*t1
        F[i,2]   += T*t2
        F[i+1,0] -= T*t0
        F[i+1,1] -= T*t1
        F[i+1,2] -= T*t2

    # Gravity, drag, damping
    for i in prange(N+1):
        pos0 = X[i,0]
        pos1 = X[i,1]
        pos2 = X[i,2]
        vel0 = V[i,0]
        vel1 = V[i,1]
        vel2 = V[i,2]
        gx,gy,gz = gravity_j2(pos0,pos1,pos2)
        F[i,0]+=m*gx
        F[i,1]+=m*gy
        F[i,2]+=m*gz

        V_atm_y = omega_mars*pos0
        vrx = vel0
        vry = vel1 - V_atm_y
        vrz = vel2
        v_rel_mag = math.sqrt(vrx*vrx+vry*vry+vrz*vrz+1e-30)
        rho_atm = atmosphere_density(pos0,pos1,pos2)
        factor = -0.5*Cd*rho_atm*v_rel_mag*A
        F[i,0]+=factor*vrx
        F[i,1]+=factor*vry
        F[i,2]+=factor*vrz

        # damping
        F[i,0]+=-damping_alpha*m*vel0
        F[i,1]+=-damping_alpha*m*vel1
        F[i,2]+=-damping_alpha*m*vel2

    return F

# Initial conditions: stable configuration
X = np.zeros((N+1,3), dtype=np.float64)
for i in range(N+1):
    X[i,0] = R_mars + i*segment_length
V = np.zeros((N+1,3), dtype=np.float64)
for i in range(N+1):
    x_pos = X[i,0]
    V[i,1] = omega_mars*x_pos

print("Initial condition set: stable elevator configuration with top anchored.")
print("At t=0, we assume top anchor suddenly removed. The cable will collapse realistically.")
sys.stdout.flush()

# Warm-up JIT
print("Warming up JIT (compute_forces) to avoid long initial stall...")
dummy_F = compute_forces(X,V,N,m,E,A,segment_length,Cd,damping_alpha,omega_mars)
print("JIT warm-up done. Starting simulation now...")
sys.stdout.flush()

save_interval = 100
frames_count = num_steps//save_interval+1
positions_saved = np.zeros((frames_count, N+1,3), dtype=np.float64)
positions_saved[0,:,:] = X.copy()

if not os.path.exists('frames'):
    os.makedirs('frames')

print("Starting dynamic simulation...")
sys.stdout.flush()
start_time = time.time()
step_print_interval = max(num_steps//10,1)
frame_idx = 1

for step in range(num_steps):
    if step % step_print_interval == 0:
        elapsed = time.time()-start_time
        perc = (step/num_steps)*100
        est_total = elapsed/(perc+1e-30)*100 if perc>0 else 0
        est_rem = est_total - elapsed if perc>0 else 0
        print(f"Step {step}/{num_steps} ({perc:.1f}%): Elapsed={elapsed:.2f}s, Est.Rem={est_rem:.2f}s")
        sys.stdout.flush()

    F_now = compute_forces(X,V,N,m,E,A,segment_length,Cd,damping_alpha,omega_mars)
    X_new = X + dt*V + (dt**2/(2*m))*F_now
    F_new = compute_forces(X_new,V,N,m,E,A,segment_length,Cd,damping_alpha,omega_mars)
    V_new = V + (dt/(2*m))*(F_now+F_new)

    X = X_new
    V = V_new

    if step % save_interval == 0 and step>0:
        positions_saved[frame_idx,:,:] = X.copy()
        frame_idx += 1
        sys.stdout.flush()

    if np.isnan(X).any() or np.isnan(V).any():
        print("WARNING: NaN encountered! Stopping.")
        sys.stdout.flush()
        break
    if np.abs(X).max()>1e12 or np.abs(V).max()>1e9:
        print("WARNING: Extreme values encountered. Stopping.")
        sys.stdout.flush()
        break

end_time = time.time()
total_time = end_time - start_time
print(f"Dynamic integration finished in {total_time:.2f}s total.")
sys.stdout.flush()

positions_saved = positions_saved[:frame_idx,:,:]

print("Rendering PNG frames...")
sys.stdout.flush()
for i in range(frame_idx):
    if i % 10 == 0:
        print(f"Rendering frame {i}/{frame_idx}")
        sys.stdout.flush()
    Xf = positions_saved[i]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal','box')
    ax.set_facecolor('black')
    circle = plt.Circle((0,0), R_mars, color='red')
    ax.add_patch(circle)
    ax.plot(Xf[:,0], Xf[:,1], 'w-', lw=1)
    max_extent = R_mars+L
    ax.set_xlim(-1.2*max_extent,1.2*max_extent)
    ax.set_ylim(-1.2*max_extent,1.2*max_extent)
    ax.set_title(f"Space Elevator Collapse - Frame {i}, Time={i*save_interval*dt:.2f}s")
    plt.savefig(f"frames/frame_{i:04d}.png")
    plt.close(fig)

print("All PNG frames saved. Creating GIF.")
sys.stdout.flush()

png_files = [f"frames/frame_{i:04d}.png" for i in range(frame_idx)]
images = []
for fname in png_files:
    img = Image.open(fname)
    images.append(img.convert('RGB'))

gif_name = "space_elevator_fall.gif"
images[0].save(gif_name,
               save_all=True, append_images=images[1:], duration=0.4, loop=0)

print(f"GIF saved as {gif_name}, 10x speed.")
print(f"Total run time: {total_time:.2f}s")
print("All done.")
sys.stdout.flush()
