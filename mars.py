import math
import numpy as np
import matplotlib
import os
import sys
import time
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit

# Set MPLCONFIGDIR to a writable directory
os.environ["MPLCONFIGDIR"] = "/tmp"
matplotlib.use('Agg')

# -----------------------------------------------------------------------------------------
# Simulation of a 37,000 km space elevator collapse on Mars.
#
# Run until top node touches planet surface or max_steps reached.
# -----------------------------------------------------------------------------------------

GM = 4.282837e13
R_mars = 3389500.0
J2 = 0.00196045
omega_mars = 7.088218e-05
M_mars = GM / 6.67430e-11

L = 3.7e7     # 37000 km
rho_cable = 1300.0
A = 1e-4
# Lower E significantly for stability
E = 1e9
Cd = 1.0

rho0_atm = 0.02
H_atm = 11100.0

N = 1000
segment_length = L/N
m = rho_cable*A*segment_length

# High damping
damping_alpha = 0.1

# dt slightly larger than tiny, but still small
dt = 0.005

max_steps = 50_000_000  # large max steps

# Long anchor removal period
anchor_removal_time = 100.0
anchor_strength = E*A*10.0  # smaller anchor strength to avoid huge forces

save_interval = 10000
max_frames = 2000

print("No fenics, no MPI, no ffmpeg.")
print(f"N={N}, segment_length={segment_length}m, mass/node={m}kg")
print(f"dt={dt}s. Running until top node hits surface or max_steps={max_steps}.")
print("Lower E, higher damping, longer anchor removal for stability.")
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

@njit(fastmath=True)
def compute_forces(X,V,N,m,E,A,segment_length,Cd,damping_alpha,omega_mars,
                   X_top_init, anchor_strength, anchor_removal_time, t):
    F = np.zeros((N+1,3), dtype=np.float64)
    # Tension
    for i in range(N):
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
    for i in range(N+1):
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

    # Gradual anchor removal at top node
    if t < anchor_removal_time:
        scale = 1.0 - t/anchor_removal_time
        ax0 = X[N,0]-X_top_init[0]
        ax1 = X[N,1]-X_top_init[1]
        ax2 = X[N,2]-X_top_init[2]
        F[N,0] += -anchor_strength*ax0*scale
        F[N,1] += -anchor_strength*ax1*scale
        F[N,2] += -anchor_strength*ax2*scale

    return F

# Initial conditions:
X = np.zeros((N+1,3), dtype=np.float64)
for i in range(N+1):
    X[i,0] = R_mars + i*segment_length
V = np.zeros((N+1,3), dtype=np.float64)
for i in range(N+1):
    x_pos = X[i,0]
    V[i,1] = omega_mars*x_pos

X_top_init = X[-1].copy()

print("Initial stable config set. Will remove anchor gradually over 100s.")
sys.stdout.flush()

print("Warming up JIT...")
F_dummy = compute_forces(X,V,N,m,E,A,segment_length,Cd,damping_alpha,omega_mars,X_top_init,anchor_strength,anchor_removal_time,0.0)
print("JIT warm-up done.")
sys.stdout.flush()

positions_saved = []
positions_saved.append(X.copy())

if not os.path.exists('frames'):
    os.makedirs('frames')

start_time = time.time()
frames_saved = 1

for step in range(max_steps):
    t = step*dt
    F_now = compute_forces(X,V,N,m,E,A,segment_length,Cd,damping_alpha,omega_mars,X_top_init,anchor_strength,anchor_removal_time,t)
    X_new = X + dt*V + (dt**2/(2*m))*F_now
    F_new = compute_forces(X_new,V,N,m,E,A,segment_length,Cd,damping_alpha,omega_mars,X_top_init,anchor_strength,anchor_removal_time,t)
    V_new = V + (dt/(2*m))*(F_now+F_new)

    X = X_new
    V = V_new

    pos_top = X[N]
    r_top = math.sqrt(pos_top[0]**2+pos_top[1]**2+pos_top[2]**2)
    if r_top <= R_mars:
        print(f"Top node hit surface at step={step}, r_top={r_top:.2f}. Done.")
        sys.stdout.flush()
        break

    if np.isnan(X).any() or np.isnan(V).any():
        print("NaN encountered, stopping.")
        sys.stdout.flush()
        break
    if np.abs(X).max()>1e12 or np.abs(V).max()>1e9:
        print("Extreme values encountered, stopping.")
        sys.stdout.flush()
        break

    # Save frames at intervals
    if step % save_interval == 0 and frames_saved<max_frames:
        positions_saved.append(X.copy())
        frames_saved+=1
        # Minimal progress print
        if frames_saved%10==0:
            elapsed = time.time()-start_time
            print(f"Step={step}, Time={t:.2f}s, Frames={frames_saved}, Elapsed={elapsed:.2f}s, r_top={r_top:.2f}")
            sys.stdout.flush()

end_time = time.time()
total_time = end_time - start_time
print(f"Simulation finished in {total_time:.2f}s, steps run={step}, frames={frames_saved}")
sys.stdout.flush()

positions_saved = np.array(positions_saved,dtype=np.float64)

print("Generating PNG frames (post-simulation)...")
sys.stdout.flush()

for i in range(len(positions_saved)):
    if i % 100 == 0:
        print(f"Rendering frame {i}/{len(positions_saved)}")
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
    sim_time = i*save_interval*dt
    ax.set_title(f"Elevator Collapse - Frame {i}, Time={sim_time:.2f}s")
    plt.savefig(f"frames/frame_{i:06d}.png")
    plt.close(fig)

print("All PNG frames saved. Creating GIF (5ms per frame).")
sys.stdout.flush()

png_files = [f"frames/frame_{i:06d}.png" for i in range(len(positions_saved))]
images = []
for fname in png_files:
    img = Image.open(fname)
    images.append(img.convert('RGB'))

gif_name = "space_elevator_fall.gif"
images[0].save(gif_name,
               save_all=True, append_images=images[1:], duration=5, loop=0)

print(f"GIF saved as {gif_name}")
print(f"Total run time: {total_time:.2f}s")
sys.stdout.flush()
