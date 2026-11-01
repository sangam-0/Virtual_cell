import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# ==========================================
# 1. BIOLOGICAL & SIMULATION PARAMETERS
# ==========================================
CELL_RADIUS = 10.0
N_MEMBRANE_SLOTS = 150

LIPID_RATIO = 0.50
CHANNEL_RATIO = 0.10
PUMP_RATIO = 0.10
RECEPTOR_RATIO = 0.30

N_LIGANDS = 40
LIGAND_SPEED = 0.12

CLUSTER_SIZE = 3
CLUSTER_ANGLE_SPAN = 0.35

N_MICROTUBULES = 12
MOTOR_SPEED = 0.08
CAPTURE_RADIUS = 1.5

INITIAL_ATP = 100
MAX_ATP = 200
N_MITOCHONDRIA = 5
ATP_PRODUCTION_RATE = 0.5
ATP_COST_PUMP = 1
ATP_COST_MOTOR = 0.2

MAX_SECOND_MESSENGER = 100.0
SIGNAL_DECAY = 0.04

TRANSCRIPTION_TIME = 50
TRANSLATION_TIME = 80
GENE_TYPES = ['receptor', 'channel', 'cargo']

SECRETORY_RATE = 0.02
KINESIN_SPEED = 0.06

TIME_STEPS = 1000

# ==========================================
# 2. INITIALIZE SYSTEMS
# ==========================================
angles = np.linspace(0, 2 * np.pi, N_MEMBRANE_SLOTS, endpoint=False)
membrane_types = np.zeros(N_MEMBRANE_SLOTS, dtype=int)
n_channels = int(N_MEMBRANE_SLOTS * CHANNEL_RATIO)
n_pumps = int(N_MEMBRANE_SLOTS * PUMP_RATIO)
n_receptors = int(N_MEMBRANE_SLOTS * RECEPTOR_RATIO)

all_indices = np.arange(N_MEMBRANE_SLOTS)
np.random.shuffle(all_indices)
membrane_types[all_indices[:n_channels]] = 1
membrane_types[all_indices[n_channels:n_channels+n_pumps]] = 2
membrane_types[all_indices[n_channels+n_pumps:n_channels+n_pumps+n_receptors]] = 3

x_lipid_out = (CELL_RADIUS + 0.3) * np.cos(angles)
y_lipid_out = (CELL_RADIUS + 0.3) * np.sin(angles)
x_lipid_in = (CELL_RADIUS - 0.3) * np.cos(angles)
y_lipid_in = (CELL_RADIUS - 0.3) * np.sin(angles)
x_mem = CELL_RADIUS * np.cos(angles)
y_mem = CELL_RADIUS * np.sin(angles)

receptor_states = np.zeros(N_MEMBRANE_SLOTS, dtype=int)
bound_ligand_map = {i: -1 for i in range(N_MEMBRANE_SLOTS)}
active_membrane_slots = np.ones(N_MEMBRANE_SLOTS, dtype=bool)

microtubule_angles = np.linspace(0, 2 * np.pi, N_MICROTUBULES, endpoint=False)
microtubules = [{'angle': ang, 'x_start': 0, 'y_start': 0,
                 'x_end': (CELL_RADIUS - 1.0) * np.cos(ang),
                 'y_end': (CELL_RADIUS - 1.0) * np.sin(ang)} for ang in microtubule_angles]

def generate_solutes(count, min_r, max_r):
    r = np.sqrt(np.random.rand(count)) * (max_r - min_r) + min_r
    theta = 2 * np.pi * np.random.rand(count)
    return r * np.cos(theta), r * np.sin(theta)

x_lig, y_lig = generate_solutes(N_LIGANDS, CELL_RADIUS + 1.5, CELL_RADIUS + 8.0)
vx_lig = np.random.normal(0, LIGAND_SPEED, N_LIGANDS)
vy_lig = np.random.normal(0, LIGAND_SPEED, N_LIGANDS)
active_ligands = np.ones(N_LIGANDS, dtype=bool)

vesicles = []
vesicles_delivered = 0
lysosomes = [{'x': 1.5 * np.cos(i * 2 * np.pi / 3), 'y': 1.5 * np.sin(i * 2 * np.pi / 3), 'radius': 1.2, 'contents': 0} for i in range(3)]

atp_level = INITIAL_ATP
mitochondria = [{'x': (3 + np.random.rand() * 4) * np.cos(i), 'y': (3 + np.random.rand() * 4) * np.sin(i), 'activity': 1.0} for i in np.linspace(0, 2*np.pi, N_MITOCHONDRIA, endpoint=False)]

second_messenger_level = 0.0
mapk_activation = 0.0

nucleus = {'x': 0, 'y': 0, 'radius': 2.0}
gene_queue = []
protein_inventory = {'receptor': 0, 'channel': 0, 'cargo': 0}
secretory_vesicles = []
exocytosis_events = 0

# Data recording arrays for scientific plotting
history_time = []
history_atp = []
history_mapk = []
history_receptors = []

# ==========================================
# 3. SIMULATION LOGIC FUNCTIONS
# ==========================================
def produce_atp():
    global atp_level
    for mito in mitochondria:
        demand_factor = max(0.2, 1.0 - (atp_level / MAX_ATP))
        production = ATP_PRODUCTION_RATE * mito['activity'] * demand_factor
        atp_level = min(MAX_ATP, atp_level + production)
        mito['activity'] = 0.5 + 0.5 * demand_factor

def consume_atp(amount):
    global atp_level
    if atp_level >= amount:
        atp_level -= amount
        return True
    return False

def process_signaling():
    global second_messenger_level, mapk_activation
    bound_count = np.sum(receptor_states == 1)
    signal_input = bound_count * 3.0
    second_messenger_level = min(MAX_SECOND_MESSENGER, second_messenger_level + signal_input)
    second_messenger_level = max(0.0, second_messenger_level - SIGNAL_DECAY * MAX_SECOND_MESSENGER)
    mapk_activation = second_messenger_level / MAX_SECOND_MESSENGER

    if mapk_activation > 0.5 and np.random.rand() < 0.25:
        gene_queue.append({'type': np.random.choice(['receptor', 'cargo']), 'stage': 'transcribing', 'timer': TRANSCRIPTION_TIME})

def process_gene_expression():
    for gene in gene_queue[:]:
        gene['timer'] -= 1
        if gene['timer'] <= 0:
            if gene['stage'] == 'transcribing':
                gene['stage'] = 'translating'
                gene['timer'] = TRANSLATION_TIME
            elif gene['stage'] == 'translating':
                protein_inventory[gene['type']] += 1
                gene_queue.remove(gene)

def replenish_membrane_proteins():
    global active_membrane_slots
    if protein_inventory['receptor'] > 0:
        empty_slots = np.where(~active_membrane_slots & (membrane_types == 3))[0]
        if len(empty_slots) > 0:
            active_membrane_slots[empty_slots[0]] = True
            protein_inventory['receptor'] -= 1

def check_binding():
    for i in range(N_LIGANDS):
        if not active_ligands[i]: continue
        r = np.sqrt(x_lig[i]**2 + y_lig[i]**2)
        if CELL_RADIUS - 0.8 < r < CELL_RADIUS + 0.8:
            theta = np.arctan2(y_lig[i], x_lig[i])
            idx = np.round((theta + np.pi) / (2 * np.pi) * N_MEMBRANE_SLOTS).astype(int) % N_MEMBRANE_SLOTS
            if membrane_types[idx] == 3 and receptor_states[idx] == 0 and active_membrane_slots[idx]:
                receptor_states[idx] = 1
                bound_ligand_map[idx] = i
                active_ligands[i] = False
                x_lig[i] = x_mem[idx]
                y_lig[i] = y_mem[idx]

def check_endocytosis():
    bound_indices = np.where((receptor_states == 1) & active_membrane_slots)[0]
    if len(bound_indices) >= CLUSTER_SIZE:
        bound_angles = np.sort(angles[bound_indices])
        for i in range(len(bound_angles)):
            span = (bound_angles[(i + CLUSTER_SIZE - 1) % len(bound_angles)] - bound_angles[i]) % (2 * np.pi)
            if span < CLUSTER_ANGLE_SPAN:
                cluster_angles = [bound_angles[(i + j) % len(bound_angles)] for j in range(CLUSTER_SIZE)]
                trigger_vesicle_formation(cluster_angles)
                return

def trigger_vesicle_formation(cluster_angles):
    cluster_indices = [np.where(np.isclose(angles, ang, atol=0.01))[0][0] for ang in cluster_angles]
    mean_theta = np.mean(cluster_angles)
    vesicle_cx = (CELL_RADIUS - 2.0) * np.cos(mean_theta)
    vesicle_cy = (CELL_RADIUS - 2.0) * np.sin(mean_theta)

    v_lig_x, v_lig_y = [], []
    for idx in cluster_indices:
        lig_idx = bound_ligand_map[idx]
        if lig_idx != -1:
            v_lig_x.append(x_lig[lig_idx] - vesicle_cx)
            v_lig_y.append(y_lig[lig_idx] - vesicle_cy)
            active_ligands[lig_idx] = False
            x_lig[lig_idx], y_lig[lig_idx] = 999, 999
        active_membrane_slots[idx] = False
        receptor_states[idx] = 0

    vesicle_angles = np.linspace(0, 2 * np.pi, 20)
    vesicles.append({
        'cx': vesicle_cx, 'cy': vesicle_cy, 'vx': 0, 'vy': 0,
        'lipids_x': 1.2 * np.cos(vesicle_angles), 
        'lipids_y': 1.2 * np.sin(vesicle_angles),
        'contents_x': np.array(v_lig_x), 'contents_y': np.array(v_lig_y),
        'captured': False
    })

def transport_vesicles():
    global vesicles_delivered
    for v in vesicles:
        if v['captured']:
            if not consume_atp(ATP_COST_MOTOR): continue
            v['cx'] += v['vx']
            v['cy'] += v['vy']
            for lyso in lysosomes:
                if np.sqrt((v['cx'] - lyso['x'])**2 + (v['cy'] - lyso['y'])**2) < lyso['radius'] + 1.2:
                    lyso['contents'] += 1
                    vesicles_delivered += 1
                    v['cx'], v['cy'] = 999, 999
                    break
        else:
            v_angle = np.arctan2(v['cy'], v['cx'])
            closest_mt = microtubules[np.argmin([np.abs((v_angle - mt['angle'] + np.pi) % (2 * np.pi) - np.pi) for mt in microtubules])]
            dist_to_mt = np.sqrt((v['cx'] - (v['cx']*np.cos(closest_mt['angle']) + v['cy']*np.sin(closest_mt['angle']))*np.cos(closest_mt['angle']))**2 +
                                 (v['cy'] - (v['cx']*np.cos(closest_mt['angle']) + v['cy']*np.sin(closest_mt['angle']))*np.sin(closest_mt['angle']))**2)
            if dist_to_mt < CAPTURE_RADIUS and np.sqrt(v['cx']**2 + v['cy']**2) > 3.0:
                v['captured'] = True
                v['vx'] = -np.cos(closest_mt['angle']) * MOTOR_SPEED
                v['vy'] = -np.sin(closest_mt['angle']) * MOTOR_SPEED

# ==========================================
# 4. SETUP MULTI-PANEL SCIENTIFIC FIGURE
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1.2, 1]})

# Left Panel: Spatial Simulation
ax1.set_xlim(-16, 16)
ax1.set_ylim(-16, 16)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title("Cellular Agent-Based Simulation", fontsize=13, fontweight='bold')

nucleus_plot = plt.Circle((0, 0), nucleus['radius'], color='darkblue', alpha=0.7)
ax1.add_patch(nucleus_plot)
mito_plot = ax1.scatter([m['x'] for m in mitochondria], [m['y'] for m in mitochondria], c='red', s=120, edgecolors='darkred', linewidths=2)
ax1.add_collection(LineCollection([[(mt['x_start'], mt['y_start']), (mt['x_end'], mt['y_end'])] for mt in microtubules], colors='gray', linewidths=1.2, alpha=0.5))
lyso_plot = ax1.scatter([l['x'] for l in lysosomes], [l['y'] for l in lysosomes], c='yellow', s=200, edgecolors='orange', marker='s')
lipid_plot = ax1.scatter(np.concatenate([x_lipid_out, x_lipid_in]), np.concatenate([y_lipid_out, y_lipid_in]), c='cyan', s=15)
mem_plot = ax1.scatter(x_mem, y_mem, c=['red' if t==3 else 'green' if t==1 else 'blue' for t in membrane_types], s=70, edgecolors='black')
lig_plot = ax1.scatter(x_lig, y_lig, c='magenta', s=35, zorder=5)

status_text = ax1.text(-15, 14, "", fontsize=9, color='black', fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

# Right Panel: Scientific Time-Series Dashboard
ax2.set_xlim(0, TIME_STEPS)
ax2.set_ylim(0, 210)
ax2.set_title("Dynamic Systems Biology Metrics", fontsize=13, fontweight='bold')
ax2.set_xlabel("Simulation Time (Frames)", fontsize=11)
ax2.set_ylabel("Parameter Level / Activation", fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)

line_atp, = ax2.plot([], [], label='ATP Level (a.u.)', color='crimson', linewidth=2)
line_mapk, = ax2.plot([], [], label='MAPK Activation (%)', color='purple', linewidth=2)
line_recept, = ax2.plot([], [], label='Bound Receptors', color='darkorange', linewidth=2)
ax2.legend(loc='upper right', framealpha=0.9)

# ==========================================
# 5. ANIMATION UPDATE LOOP
# ==========================================
def update(frame):
    global x_lig, y_lig, vx_lig, vy_lig

    produce_atp()
    process_signaling()
    process_gene_expression()
    replenish_membrane_proteins()

    # Update Ligands
    for i in range(N_LIGANDS):
        if active_ligands[i]:
            x_lig[i] += vx_lig[i]
            y_lig[i] += vy_lig[i]
            r = np.sqrt(x_lig[i]**2 + y_lig[i]**2)
            if r > CELL_RADIUS + 8.0 or r < CELL_RADIUS + 0.8:
                vx_lig[i] *= -1; vy_lig[i] *= -1
            vx_lig[i] += np.random.normal(0, 0.02)
            vy_lig[i] += np.random.normal(0, 0.02)

    check_binding()
    check_endocytosis()
    transport_vesicles()

    # Update Left Panel Plots
    lig_plot.set_offsets(np.c_[x_lig[active_ligands], y_lig[active_ligands]])

    mem_colors = []
    for i in range(N_MEMBRANE_SLOTS):
        if not active_membrane_slots[i]: mem_colors.append('white')
        elif membrane_types[i] == 1: mem_colors.append('green')
        elif membrane_types[i] == 2: mem_colors.append('blue')
        else: mem_colors.append('darkred' if receptor_states[i] == 1 else 'red')
    mem_plot.set_color(mem_colors)

    # Render Vesicles dynamically
    for child in ax1.get_children():
        if hasattr(child, 'is_vesicle'): child.remove()
    for v in vesicles:
        if v['cx'] < 900:
            scat_v = ax1.scatter(v['lipids_x'] + v['cx'], v['lipids_y'] + v['cy'], c='orange' if v['captured'] else 'cyan', s=25, edgecolors='black')
            scat_v.is_vesicle = True
            if len(v['contents_x']) > 0:
                scat_c = ax1.scatter(v['contents_x'] + v['cx'], v['contents_y'] + v['cy'], c='magenta', s=30, zorder=5)
                scat_c.is_vesicle = True

    nucleus_plot.set_color('purple' if mapk_activation > 0.4 else 'darkblue')

    status_text.set_text(
        f"ATP: {atp_level:.0f} / {MAX_ATP}\n"
        f"MAPK: {mapk_activation*100:.1f}%\n"
        f"Active Genes: {len(gene_queue)}\n"
        f"Proteins: R:{protein_inventory['receptor']} C:{protein_inventory['channel']} Ca:{protein_inventory['cargo']}"
    )

    # Update Right Panel Data Curves
    history_time.append(frame)
    history_atp.append(atp_level)
    history_mapk.append(mapk_activation * 100)
    history_receptors.append(np.sum(receptor_states == 1) * 10)

    line_atp.set_data(history_time, history_atp)
    line_mapk.set_data(history_time, history_mapk)
    line_recept.set_data(history_time, history_receptors)

    return lig_plot, mem_plot, status_text, line_atp, line_mapk, line_recept

print("Running Scientific Animation Dashboard...")
ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, interval=40, blit=False)

plt.tight_layout()
plt.show()
