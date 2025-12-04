import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec

# --- 0. 绘图设置 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False 
ls = LightSource(azdeg=315, altdeg=45)

# --- 1. 物理参数 ---
length = 1.5
width_base = 0.4      
width_extended = 1.2  

# 动作角度
max_bend_deg = 40.0 
landing_bend_deg = 15.0

# 动画帧数
frames_bend = 40
frames_spin = 80
frames_unbend = 40
frames_total = frames_bend + frames_spin + frames_unbend + 20

# === 自由落体参数 (Free Fall Physics) ===
# 我们假设整个翻转过程对应真实的 0.6 秒 (猫从约 1.8米处跌落)
total_real_time = 0.6 
dt_real = total_real_time / (frames_bend + frames_spin + frames_unbend) # 每帧对应的真实时间
g = 9.8 # 重力加速度

# 计算初始高度，使得 t_end 时 height = 0
# H = 1/2 * g * t^2
initial_height = 0.5 * g * (total_real_time ** 2) 
# 加上一点余量，让猫最后悬停在地面上方一点点
initial_height += 0.5 

# 状态存储
history = {
    'time': [],
    'Lx_f': [], 'Lx_b': [], 'Lx_tot': [],
    'Ly_f': [], 'Ly_b': [], 'Ly_tot': [],
    'Lz_f': [], 'Lz_b': [], 'Lz_tot': [],
    'Height': [] # 记录高度
}

current_global_q = R.from_quat([0, 0, 0, 1]) 

# --- 2. 物理核心引擎 ---

def get_body_inertia(w, l, m=1.0):
    h = w
    Ixx = (m/12.0) * (w**2 + h**2)       
    Iyy = (m/12.0) * (l**2 + h**2)       
    Izz = (m/12.0) * (l**2 + w**2)       
    return np.diag([Ixx, Iyy, Izz])

def compute_frame_physics(frame):
    # --- A. 形状运动学 ---
    beta = 0.0; d_beta = 0.0; phi = 0.0; d_phi = 0.0
    current_w_physics = width_base
    
    beta_max = np.radians(max_bend_deg)
    beta_land = np.radians(landing_bend_deg)
    
    if frame < frames_bend:
        p = frame / frames_bend
        p_smooth = (1 - np.cos(p * np.pi)) / 2 
        beta = p_smooth * beta_max
        if frame > 0: d_beta = (beta_max/frames_bend) * np.sin(p*np.pi) * 1.5 
        
    elif frame < frames_bend + frames_spin:
        beta = beta_max
        p = (frame - frames_bend) / frames_spin
        phi = p * 2 * np.pi 
        d_phi = 2 * np.pi / frames_spin * 30.0
        ext = np.sin(p * np.pi) 
        current_w_physics = width_base + (width_extended - width_base) * ext
        
    elif frame < frames_bend + frames_spin + frames_unbend:
        p = (frame - (frames_bend + frames_spin)) / frames_unbend
        p_smooth = (1 - np.cos(p * np.pi)) / 2
        beta = beta_max - (beta_max - beta_land) * p_smooth
        d_beta = - ((beta_max - beta_land)/frames_unbend) * np.sin(p*np.pi) * 1.5
        phi = 2 * np.pi
    else:
        beta = beta_land
        phi = 2 * np.pi

    # --- B. 局部坐标系 ---
    R_bend_f = R.from_euler('y', -beta, degrees=False)
    R_bend_b = R.from_euler('y', +beta, degrees=False)
    R_spin_f = R.from_euler('x', phi, degrees=False)
    R_spin_b = R.from_euler('x', phi, degrees=False)
    
    R_f = R_bend_f * R_spin_f
    R_b = R_bend_b * R_spin_b
    
    # --- C. 惯性与动量求解 ---
    I_local = get_body_inertia(current_w_physics, length)
    I_f_global = R_f.as_matrix() @ I_local @ R_f.as_matrix().T
    I_b_global = R_b.as_matrix() @ I_local @ R_b.as_matrix().T
    I_sys = I_f_global + I_b_global
    
    w_f_rel = R_f.as_matrix() @ np.array([d_phi, -d_beta, 0]) 
    w_b_rel = R_b.as_matrix() @ np.array([d_phi, d_beta, 0])
    
    L_int = I_f_global @ w_f_rel + I_b_global @ w_b_rel
    
    I_sys_inv = np.linalg.inv(I_sys)
    w_base = - I_sys_inv @ L_int
    
    L_f_abs = I_f_global @ (w_base + w_f_rel)
    L_b_abs = I_b_global @ (w_base + w_b_rel)
    L_tot_abs = L_f_abs + L_b_abs 
    
    return w_base, beta, phi, current_w_physics, L_f_abs, L_b_abs, L_tot_abs

# --- 3. 几何生成 ---
def generate_box(w, l, twist=0, bend=0, direction=1):
    x = np.linspace(0, l * direction, 2)
    angles = np.linspace(np.pi/4, 2*np.pi+np.pi/4, 5)
    yy = w * np.cos(angles); zz = w * np.sin(angles)
    xx_grid, _ = np.meshgrid(x, angles)
    y_grid = np.tile(yy[:, np.newaxis], (1, 2))
    z_grid = np.tile(zz[:, np.newaxis], (1, 2))
    points = np.stack([xx_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=1)
    
    r_twist = R.from_euler('x', twist, degrees=False)
    r_bend = R.from_euler('y', bend, degrees=False)
    r_local = r_bend * r_twist
    
    points[:, 0] += 0.2 * direction
    return r_local.apply(points), xx_grid.shape

# --- 4. 动画循环 ---

def update(frame):
    global current_global_q
    
    # 动画逻辑时间步长 (用于旋转积分)
    dt_sim = 1.0/30.0 
    
    w_base, beta, phi, w_physics, Lf, Lb, Lt = compute_frame_physics(frame)
    
    if frame > 0: 
        current_global_q = R.from_rotvec(w_base * dt_sim) * current_global_q
    
    # === 计算下落高度 (Free Fall) ===
    # t_real 是真实物理时间
    t_real = frame * dt_real
    # h = h0 - 0.5 * g * t^2
    current_height = initial_height - 0.5 * g * (t_real**2)
    
    # 地面限制 (不钻地)
    if current_height < 0.4: current_height = 0.4 # 保持最后落地高度
    
    # 记录
    history['time'].append(frame)
    history['Lx_f'].append(Lf[0]); history['Lx_b'].append(Lb[0]); history['Lx_tot'].append(Lt[0])
    history['Ly_f'].append(Lf[1]); history['Ly_b'].append(Lb[1]); history['Ly_tot'].append(Lt[1])
    history['Lz_f'].append(Lf[2]); history['Lz_b'].append(Lb[2]); history['Lz_tot'].append(Lt[2])
    history['Height'].append(current_height)
    if len(history['time']) > 150:
        for k in history: history[k].pop(0)
        
    # === 绘图 ===
    ax_3d.clear()
    ax_lx.clear(); ax_ly.clear(); ax_lz.clear()
    
    # --- 3D View ---
    ax_3d.set_proj_type('persp'); ax_3d.set_axis_off()
    ax_3d.set_xlim(-3.5, 3.5); ax_3d.set_ylim(-3.5, 3.5); ax_3d.set_zlim(-3.5, 3.5)
    
    # [关键] 动态地面
    # 地面相对于猫(0,0,0)的位置就是 -current_height
    # 我们画一个 Z = -current_height 的平面
    ground_z = -current_height
    
    # 地面网格
    xx, yy = np.meshgrid(np.linspace(-4,4,10), np.linspace(-4,4,10))
    
    # 根据高度改变地面透明度/颜色，增强逼近感
    # 越高越淡，越低越深
    ground_alpha = 0.1 + 0.3 * (1 - min(1, current_height/initial_height))
    ax_3d.plot_wireframe(xx, yy, np.full_like(xx, ground_z), color='#555555', alpha=ground_alpha, linewidth=1)
    
    # 投影阴影 (Shadow)
    # 在地面上画一个简单的黑色矩形阴影，随高度缩小? 或者简单点，投影猫的形状
    # 为了性能，这里简单画一个随高度变大变清晰的阴影点
    shadow_s = 1.0 + (current_height * 0.1) # 越高越模糊(大)
    shadow_a = 0.5 * (1 / (current_height + 0.5)) # 越低越黑
    ax_3d.plot([0], [0], [ground_z], 'o', color='black', alpha=min(0.8, shadow_a), markersize=30*shadow_s, zorder=0)

    # 几何体 (猫始终在原点)
    vis_width = width_base 
    pts_f, shape = generate_box(vis_width, length, twist=phi, bend=-beta, direction=1)
    pts_b, _     = generate_box(vis_width, length, twist=phi, bend=beta, direction=-1)
    
    Xf, Yf, Zf = current_global_q.apply(pts_f).T.reshape((3, *shape))
    Xb, Yb, Zb = current_global_q.apply(pts_b).T.reshape((3, *shape))
    
    rgb_f = ls.shade(Zf, cmap=plt.cm.Reds, vert_exag=0.1, blend_mode='soft')
    rgb_b = ls.shade(Zb, cmap=plt.cm.Blues, vert_exag=0.1, blend_mode='soft')
    
    ax_3d.plot_surface(Xf, Yf, Zf, facecolors=rgb_f, shade=False, alpha=0.9, edgecolor='k', linewidth=0.5)
    ax_3d.plot_surface(Xb, Yb, Zb, facecolors=rgb_b, shade=False, alpha=0.9, edgecolor='k', linewidth=0.5)
    
    ss = current_global_q.apply([0.2,0,0]); se = current_global_q.apply([-0.2,0,0])
    ax_3d.plot([ss[0], se[0]], [ss[1], se[1]], [ss[2], se[2]], 'k-', lw=4)

    # 显示高度计
    ax_3d.text2D(0.05, 0.90, f"Altitude: {current_height:.2f} m", transform=ax_3d.transAxes, fontsize=12, color='blue')
    
    status = "Falling & Reacting"
    if current_height <= 0.5: status = "IMPACT / LANDING!"
    ax_3d.text2D(0.05, 0.95, status, transform=ax_3d.transAxes, fontsize=14, fontweight='bold')

    # --- Plots ---
    def plot_component(ax, f_data, b_data, t_data, title, ylabel):
        ax.set_title(title, fontsize=10, pad=2)
        ax.set_ylabel(ylabel, fontsize=9)
        all_d = np.concatenate([f_data, b_data])
        ymax = np.max(np.abs(all_d)) if len(all_d) > 0 else 1.0
        if ymax < 0.1: ymax = 0.5
        ax.set_ylim(-ymax*1.2, ymax*1.2)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([]) 
        ax.plot(f_data, 'r-', lw=1.5, alpha=0.7, label='Front')
        ax.plot(b_data, 'b-', lw=1.5, alpha=0.7, label='Back')
        ax.plot(t_data, 'g-', lw=2.5, label='Total')
        
    plot_component(ax_lx, history['Lx_f'], history['Lx_b'], history['Lx_tot'], "Angular Momentum X (Roll)", "$L_x$")
    ax_lx.legend(loc='upper right', fontsize=7, frameon=False, ncol=3)
    plot_component(ax_ly, history['Ly_f'], history['Ly_b'], history['Ly_tot'], "Angular Momentum Y (Pitch)", "$L_y$")
    plot_component(ax_lz, history['Lz_f'], history['Lz_b'], history['Lz_tot'], "Angular Momentum Z (Yaw)", "$L_z$")
    
# --- 5. 运行 ---

fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.5, 1])

ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_lx = fig.add_subplot(gs[0, 1])
ax_ly = fig.add_subplot(gs[1, 1])
ax_lz = fig.add_subplot(gs[2, 1])

# fps=30
anim = FuncAnimation(fig, update, frames=frames_total, interval=33, repeat=False)
anim.save("cat_reflex1.gif", writer=PillowWriter(fps=30))
plt.tight_layout()
plt.show()