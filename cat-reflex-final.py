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

# 动画帧数配置
frames_bend = 40
frames_spin = 80
frames_unbend = 40
frames_total = frames_bend + frames_spin + frames_unbend + 20

# === 自由落体与时间参数 ===
total_real_time = 0.6  # 真实物理时间 0.6秒
# 每一帧代表的真实物理时长 (dt)
dt_real = total_real_time / (frames_bend + frames_spin + frames_unbend) 
g = 9.8 

# 初始高度
initial_height = 0.5 * g * (total_real_time ** 2) + 0.5 

# 状态存储
history = {
    'time': [],
    'Lx_f': [], 'Lx_b': [], 'Lx_tot': [],
    'Ly_f': [], 'Ly_b': [], 'Ly_tot': [],
    'Lz_f': [], 'Lz_b': [], 'Lz_tot': [],
    'Height': []
}

current_global_q = R.from_quat([0, 0, 0, 1]) 

# --- 2. 物理核心引擎 ---

def get_body_inertia(w, l, m=1.0):
    # 简单的长方体转动惯量近似
    h = w
    Ixx = (m/12.0) * (w**2 + h**2)       
    Iyy = (m/12.0) * (l**2 + h**2)       
    Izz = (m/12.0) * (l**2 + w**2)       
    return np.diag([Ixx, Iyy, Izz])

def compute_frame_physics(frame):
    # --- A. 形状运动学 (计算位置和速度) ---
    beta = 0.0; d_beta = 0.0; phi = 0.0; d_phi = 0.0
    current_w_physics = width_base
    
    beta_max = np.radians(max_bend_deg)
    beta_land = np.radians(landing_bend_deg)
    
    # 辅助导数：p_smooth = (1 - cos(p*pi))/2 
    # derivative d(p_smooth)/dp = 0.5 * pi * sin(p*pi)
    
    if frame < frames_bend:
        # Phase 1: Bend
        p = frame / frames_bend
        # 位置
        beta = beta_max * (1 - np.cos(p * np.pi)) / 2
        # 速度 (rad/frame)
        beta_dot_frame = beta_max * 0.5 * np.pi * np.sin(p * np.pi) / frames_bend
        # 速度 (rad/s) -> [FIX: Normalize by dt_real]
        d_beta = beta_dot_frame / dt_real
        
    elif frame < frames_bend + frames_spin:
        # Phase 2: Spin (Twist)
        beta = beta_max
        p = (frame - frames_bend) / frames_spin
        
        # Twist Angle
        phi = p * 2 * np.pi 
        # Twist Velocity (rad/s) -> [FIX: Normalize by dt_real]
        d_phi = (2 * np.pi / frames_spin) / dt_real
        
        # Leg Extension (影响惯量)
        ext = np.sin(p * np.pi) 
        current_w_physics = width_base + (width_extended - width_base) * ext
        
    elif frame < frames_bend + frames_spin + frames_unbend:
        # Phase 3: Unbend
        p = (frame - (frames_bend + frames_spin)) / frames_unbend
        p_smooth = (1 - np.cos(p * np.pi)) / 2
        
        beta = beta_max - (beta_max - beta_land) * p_smooth
        
        beta_dot_frame = - (beta_max - beta_land) * 0.5 * np.pi * np.sin(p * np.pi) / frames_unbend
        d_beta = beta_dot_frame / dt_real
        
        phi = 2 * np.pi
    else:
        # Phase 4: Landing
        beta = beta_land
        phi = 2 * np.pi
        d_beta = 0.0
        d_phi = 0.0

    # --- B. 坐标系构建 ---
    # Bend: 绕 Y 轴
    R_bend_f = R.from_euler('y', -beta, degrees=False)
    R_bend_b = R.from_euler('y', +beta, degrees=False)
    # Spin: 绕 X 轴
    R_spin_f = R.from_euler('x', phi, degrees=False)
    R_spin_b = R.from_euler('x', phi, degrees=False)
    
    R_f = R_bend_f * R_spin_f
    R_b = R_bend_b * R_spin_b
    
    # --- C. 动力学求解 (Corrected Vector Math + Units) ---
    I_local = get_body_inertia(current_w_physics, length)
    
    # 1. Global Inertia Tensor
    I_f_global = R_f.as_matrix() @ I_local @ R_f.as_matrix().T
    I_b_global = R_b.as_matrix() @ I_local @ R_b.as_matrix().T
    I_sys = I_f_global + I_b_global
    
    # 2. Angular Velocity Vectors (rad/s)
    # Twist 是绕局部 X 轴，需要变换到全局 (但在 Bend 之后)
    w_twist_f_global = R_bend_f.apply(np.array([d_phi, 0, 0]))
    w_twist_b_global = R_bend_b.apply(np.array([d_phi, 0, 0]))
    
    # Bend 是绕全局 Y (脊柱轴)
    w_bend_f_global = np.array([0, -d_beta, 0])
    w_bend_b_global = np.array([0, d_beta, 0])
    
    w_f_rel = w_bend_f_global + w_twist_f_global
    w_b_rel = w_bend_b_global + w_twist_b_global
    
    # 3. Solve Conservation Law
    # L_total = I_sys * w_base + (I_f * w_f_rel + I_b * w_b_rel) = 0
    L_int = I_f_global @ w_f_rel + I_b_global @ w_b_rel
    
    I_sys_inv = np.linalg.inv(I_sys)
    w_base = - I_sys_inv @ L_int # Result is in rad/s
    
    # 4. Verify Momentum (For Plotting)
    L_f_abs = I_f_global @ (w_base + w_f_rel)
    L_b_abs = I_b_global @ (w_base + w_b_rel)
    L_tot_abs = L_f_abs + L_b_abs 
    
    return w_base, beta, phi, current_w_physics, L_f_abs, L_b_abs, L_tot_abs

# --- 3. 几何生成 (Visualization) ---
def generate_box(w, l, twist=0, bend=0, direction=1):
    # 稍微加宽一点显示宽度，看起来更像猫
    vis_w = 0.5 
    vis_h = 0.5
    
    x = np.linspace(0, l * direction, 2)
    # 创建一个方形截面
    pts = []
    # Front face
    pts.append([0, vis_w/2, vis_h/2])
    pts.append([0, -vis_w/2, vis_h/2])
    pts.append([0, -vis_w/2, -vis_h/2])
    pts.append([0, vis_w/2, -vis_h/2])
    # Back face
    pts.append([l*direction, vis_w/2, vis_h/2])
    pts.append([l*direction, -vis_w/2, vis_h/2])
    pts.append([l*direction, -vis_w/2, -vis_h/2])
    pts.append([l*direction, vis_w/2, -vis_h/2])
    
    points = np.array(pts)
    
    # 简单的网格化用于绘图 (Surface)
    # 这里为了代码简洁，用简单的参数化圆柱体近似长方体绘制，或者直接变换点
    # 为了保持和原代码一致的风格，我们沿用参数化生成：
    
    angles = np.linspace(np.pi/4, 2*np.pi+np.pi/4, 5)
    yy = vis_w * 0.8 * np.cos(angles)
    zz = vis_h * 0.8 * np.sin(angles)
    xx_grid, _ = np.meshgrid(x, angles)
    y_grid = np.tile(yy[:, np.newaxis], (1, 2))
    z_grid = np.tile(zz[:, np.newaxis], (1, 2))
    grid_points = np.stack([xx_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=1)
    
    r_twist = R.from_euler('x', twist, degrees=False)
    r_bend = R.from_euler('y', bend, degrees=False)
    r_local = r_bend * r_twist
    
    grid_points[:, 0] += 0.2 * direction # Gap offset
    transformed_points = r_local.apply(grid_points)
    
    return transformed_points, xx_grid.shape

# --- 4. 动画循环 ---

def update(frame):
    global current_global_q
    
    # 计算物理 (返回的是 rad/s)
    w_base, beta, phi, w_physics, Lf, Lb, Lt = compute_frame_physics(frame)
    
    # 积分: q_new = q_old + 0.5 * w * q_old * dt
    # 使用 Scipy 的 exp map: R(w * dt)
    if frame > 0: 
        # [FIX] 使用 dt_real 进行积分，匹配物理单位
        angle_step = w_base * dt_real 
        current_global_q = R.from_rotvec(angle_step) * current_global_q
    
    # 下落位移
    t_real = frame * dt_real
    current_height = initial_height - 0.5 * g * (t_real**2)
    if current_height < 0.4: current_height = 0.4
    
    # 记录数据
    history['time'].append(t_real)
    history['Lx_f'].append(Lf[0]); history['Lx_b'].append(Lb[0]); history['Lx_tot'].append(Lt[0])
    history['Ly_f'].append(Lf[1]); history['Ly_b'].append(Lb[1]); history['Ly_tot'].append(Lt[1])
    history['Lz_f'].append(Lf[2]); history['Lz_b'].append(Lb[2]); history['Lz_tot'].append(Lt[2])
    history['Height'].append(current_height)
    
    # 滚动窗口
    if len(history['time']) > frames_total: # 保留全长以便观察
        for k in history: history[k].pop(0)
        
    # === 绘图 ===
    ax_3d.clear()
    ax_lx.clear(); ax_ly.clear(); ax_lz.clear()
    
    # 3D 设置
    ax_3d.set_proj_type('persp'); ax_3d.set_axis_off()
    ax_3d.set_xlim(-2.5, 2.5); ax_3d.set_ylim(-2.5, 2.5); ax_3d.set_zlim(-2.5, 2.5)
    
    # 地面
    ground_z = -current_height
    xx, yy = np.meshgrid(np.linspace(-3,3,8), np.linspace(-3,3,8))
    ground_alpha = 0.15 + 0.4 * (1 - min(1, current_height/initial_height))
    ax_3d.plot_wireframe(xx, yy, np.full_like(xx, ground_z), color='#444444', alpha=ground_alpha, lw=0.8)
    # 阴影
    shadow_s = 1.0 + (current_height * 0.15)
    shadow_a = 0.6 * (1 / (current_height + 0.6))
    ax_3d.plot([0], [0], [ground_z], 'o', color='black', alpha=min(0.8, shadow_a), markersize=25*shadow_s, zorder=0)

    # 猫
    vis_width = width_base 
    pts_f, shape = generate_box(vis_width, length, twist=phi, bend=-beta, direction=1)
    pts_b, _     = generate_box(vis_width, length, twist=phi, bend=beta, direction=-1)
    
    Xf, Yf, Zf = current_global_q.apply(pts_f).T.reshape((3, *shape))
    Xb, Yb, Zb = current_global_q.apply(pts_b).T.reshape((3, *shape))
    
    rgb_f = ls.shade(Zf, cmap=plt.cm.Reds, vert_exag=0.1, blend_mode='soft')
    rgb_b = ls.shade(Zb, cmap=plt.cm.Blues, vert_exag=0.1, blend_mode='soft')
    
    ax_3d.plot_surface(Xf, Yf, Zf, facecolors=rgb_f, shade=False, alpha=0.95, edgecolor='k', linewidth=0.3)
    ax_3d.plot_surface(Xb, Yb, Zb, facecolors=rgb_b, shade=False, alpha=0.95, edgecolor='k', linewidth=0.3)
    
    # 脊柱连接线
    ss = current_global_q.apply([0.2,0,0]); se = current_global_q.apply([-0.2,0,0])
    ax_3d.plot([ss[0], se[0]], [ss[1], se[1]], [ss[2], se[2]], 'k-', lw=3)

    ax_3d.text2D(0.05, 0.92, f"Alt: {current_height:.2f} m", transform=ax_3d.transAxes, color='blue', fontweight='bold')

    # === Plot Curves ===
    def plot_component(ax, f_data, b_data, t_data, title, ylabel):
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_ylabel(ylabel, fontsize=8)
        
        # 动态范围: 保证看到波动
        all_d = np.concatenate([f_data, b_data])
        ymax = np.max(np.abs(all_d)) if len(all_d) > 0 else 0.1
        if ymax < 0.1: ymax = 0.5 # 只有在极小值时才强制默认
        else: ymax *= 1.2         # 否则留出 20% 余量
            
        ax.set_ylim(-ymax, ymax)
        ax.grid(True, alpha=0.3, ls='--')
        ax.set_xticks([]) # Remove time ticks for cleaner look
        
        ax.plot(f_data, color='red', lw=1.5, alpha=0.8, label='Front')
        ax.plot(b_data, color='blue', lw=1.5, alpha=0.8, label='Back')
        ax.plot(t_data, color='green', lw=2.0, alpha=0.9, label='Total')

    plot_component(ax_lx, history['Lx_f'], history['Lx_b'], history['Lx_tot'], "Roll ($L_x$)", "$L_x$")
    ax_lx.legend(loc='upper right', fontsize=6, ncol=3, framealpha=0.5)
    plot_component(ax_ly, history['Ly_f'], history['Ly_b'], history['Ly_tot'], "Pitch ($L_y$)", "$L_y$")
    plot_component(ax_lz, history['Lz_f'], history['Lz_b'], history['Lz_tot'], "Yaw ($L_z$)", "$L_z$")

# --- 5. 运行 ---
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.6, 1], wspace=0.15, hspace=0.25)

ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_lx = fig.add_subplot(gs[0, 1])
ax_ly = fig.add_subplot(gs[1, 1])
ax_lz = fig.add_subplot(gs[2, 1])

# 为了 GIF 导出速度，这里 interval 设为 30ms
anim = FuncAnimation(fig, update, frames=frames_total, interval=30, repeat=False)
anim.save("cat_reflex_physics_verified.gif", writer=PillowWriter(fps=30))
plt.show()