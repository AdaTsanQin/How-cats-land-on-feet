import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.gridspec as gridspec  # <--- 必须包含这行，否则会报错
from matplotlib.colors import LightSource
import platform

# --- 0. 设置 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14  
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

# 设置光照 (右上方光源)
ls = LightSource(azdeg=315, altdeg=45)

# --- 1. 物理参数 ---
I_0 = 1.0             
ratio_shrink = 0.5  
ratio_expand = 1.5  
bend_angle_max = 90 

# 身体间隙 (Gap)
gap_half = 0.4 

# 时间轴
frames_bend = 40      
frames_wait1 = 15
frames_twist1 = 70    
frames_wait2 = 15
frames_twist2 = 70    
frames_wait3 = 15
frames_unbend = 50    
frames_end = 30

state = {
    'phi_front': 0.0,
    'phi_back': 0.0,
    'bend': 0.0,
    'L_history': [], 'Lf_history': [], 'Lb_history': [], 'frame_history': []
}

# --- 2. 几何生成函数 ---

def generate_square_beam(width, length, twist=0, direction=1):
    """生成长方体 (解决旋转错觉)"""
    x = np.linspace(0, length * direction, 2)
    theta = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 5) 
    theta_grid, x_grid = np.meshgrid(theta, x)
    
    y_grid = width * np.cos(theta_grid + twist)
    z_grid = width * np.sin(theta_grid + twist)
    
    # 刻线
    stripe_y = width * np.cos(twist + np.pi/4)
    stripe_z = width * np.sin(twist + np.pi/4)
    stripe = (x, np.array([stripe_y, stripe_y]), np.array([stripe_z, stripe_z]))
    
    return x_grid, y_grid, z_grid, stripe

def rotate_z_global(x, y, z, angle_deg):
    """绕Z轴旋转 (折叠)"""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    x_new = x * c - y * s
    y_new = x * s + y * c
    return x_new, y_new, z

# --- 3. 动画核心逻辑 ---

def update(frame):
    ax_3d.clear()
    ax_L.clear()
    
    # 透视投影 (关键：消除视觉错觉)
    ax_3d.set_proj_type('persp')
    
    limit = 3.0
    ax_3d.set_xlim(-limit, limit); ax_3d.set_ylim(-limit, limit); ax_3d.set_zlim(-limit, limit)
    ax_3d.set_axis_off()
    
    # --- 阶段计算 ---
    t1 = frames_bend
    t2 = t1 + frames_wait1
    t3 = t2 + frames_twist1
    t4 = t3 + frames_wait2
    t5 = t4 + frames_twist2
    t6 = t5 + frames_wait3
    t7 = t6 + frames_unbend
    
    w_front = 0.0
    w_back = 0.0
    I_f_curr = I_0
    I_b_curr = I_0
    
    status_title = ""
    physics_explanation = "" 

    if frame < t1:
        status_title = "Phase A: Spine Bending"
        physics_explanation = "Bending the central spine axis."
        progress = frame / frames_bend
        state['bend'] = progress * (bend_angle_max / 2)
        
    elif frame < t2:
        status_title = "Preparation..."
        
    elif frame < t3:
        status_title = "Phase B: Front Twist"
        physics_explanation = "Front (Red) Retracts -> Fast Rotation\nBack (Blue) Extends -> Slow Counter-rotation"
        I_f_curr = ratio_shrink * I_0
        I_b_curr = ratio_expand * I_0
        delta_phi_f = np.pi / frames_twist1
        delta_phi_b = -(I_f_curr / I_b_curr) * delta_phi_f
        state['phi_front'] += delta_phi_f
        state['phi_back'] += delta_phi_b
        w_front = delta_phi_f
        w_back = delta_phi_b

    elif frame < t4:
        status_title = "Swapping Inertia..."
        physics_explanation = "Inertia Swap: Red expands, Blue shrinks."

    elif frame < t5:
        status_title = "Phase C: Back Twist"
        physics_explanation = "Back (Blue) Retracts -> Fast Rotation\nFront (Red) Extends -> Slow Counter-rotation"
        I_f_curr = ratio_expand * I_0
        I_b_curr = ratio_shrink * I_0
        delta_phi_b = np.pi / frames_twist2
        delta_phi_f = -(I_b_curr / I_f_curr) * delta_phi_b
        state['phi_front'] += delta_phi_f
        state['phi_back'] += delta_phi_b
        w_front = delta_phi_f
        w_back = delta_phi_b

    elif frame < t6:
        status_title = "Preparation..."

    elif frame < t7:
        status_title = "Phase D: Re-alignment"
        physics_explanation = "Unbending the spine axis."
        progress = (frame - t6) / frames_unbend
        max_b = bend_angle_max / 2
        state['bend'] = max_b * (1 - progress)
        
    else:
        status_title = "Goal Achieved"
        physics_explanation = "Body horizontal. Spine straight."
        state['bend'] = 0

    # 数据记录
    L_f = I_f_curr * w_front * 100 
    L_b = I_b_curr * w_back * 100
    L_total = L_f + L_b
    state['frame_history'].append(frame)
    state['Lf_history'].append(L_f)
    state['Lb_history'].append(L_b)
    state['L_history'].append(L_total)
    if len(state['frame_history']) > 200:
        for k in ['frame_history','Lf_history','Lb_history','L_history']: state[k].pop(0)

    # --- 绘制 3D 模型 ---
    
    # 1. 地面网格
    xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
    zz = np.full_like(xx, -2.5)
    ax_3d.plot_wireframe(xx, yy, zz, color='gray', alpha=0.2, linewidth=1)

    # 2. 生成几何体 (应用间隙)
    # 前半身 (红)
    Xf, Yf, Zf, Sf = generate_square_beam(0.6, 1.5, twist=state['phi_front'], direction=1)
    Xf += gap_half
    Sf = (Sf[0] + gap_half, Sf[1], Sf[2])
    
    # 后半身 (蓝)
    Xb, Yb, Zb, Sb = generate_square_beam(0.6, 1.5, twist=state['phi_back'], direction=-1)
    Xb -= gap_half
    Sb = (Sb[0] - gap_half, Sb[1], Sb[2])
    
    # 3. 脊柱连接线
    spine_x = np.array([-gap_half, gap_half])
    spine_y = np.array([0, 0])
    spine_z = np.array([0, 0])

    # 4. 应用折叠旋转
    Xf, Yf, Zf = rotate_z_global(Xf, Yf, Zf, state['bend'])
    sfx, sfy, sfz = rotate_z_global(Sf[0], Sf[1], Sf[2], state['bend'])
    
    Xb, Yb, Zb = rotate_z_global(Xb, Yb, Zb, -state['bend'])
    sbx, sby, sbz = rotate_z_global(Sb[0], Sb[1], Sb[2], -state['bend'])
    
    # 脊柱旋转 (前半段和后半段分别旋转)
    sp_f_x, sp_f_y, sp_f_z = rotate_z_global(np.array([0, gap_half]), np.array([0,0]), np.array([0,0]), state['bend'])
    sp_b_x, sp_b_y, sp_b_z = rotate_z_global(np.array([-gap_half, 0]), np.array([0,0]), np.array([0,0]), -state['bend'])
    spine_x_all = np.concatenate([sp_b_x, sp_f_x])
    spine_y_all = np.concatenate([sp_b_y, sp_f_y])
    spine_z_all = np.concatenate([sp_b_z, sp_f_z])

    # 5. 渲染
    rgb_f = ls.shade(Zf, cmap=plt.cm.Reds, vert_exag=0.1, blend_mode='soft')
    rgb_b = ls.shade(Zb, cmap=plt.cm.Blues, vert_exag=0.1, blend_mode='soft')

    ax_3d.plot_surface(Xf, Yf, Zf, facecolors=rgb_f, shade=False, alpha=0.9, 
                       edgecolor='black', linewidth=1, rstride=1, cstride=1)
    
    ax_3d.plot_surface(Xb, Yb, Zb, facecolors=rgb_b, shade=False, alpha=0.9, 
                       edgecolor='black', linewidth=1, rstride=1, cstride=1)

    # 绘制脊柱粗线
    ax_3d.plot(spine_x_all, spine_y_all, spine_z_all, color='black', linewidth=6, solid_capstyle='round')

    ax_3d.text2D(0.02, 0.95, status_title, transform=ax_3d.transAxes, fontsize=20, fontweight='bold', color='#333')
    ax_3d.view_init(elev=60, azim=-80)

    # --- 绘制角动量图 ---
    ax_L.set_title("Zero Angular Momentum Conservation ($L_{total} = 0$)", fontsize=16, fontweight='bold')
    ax_L.set_ylim(-12, 12)
    ax_L.set_xlim(max(0, frame-200), frame+10)
    ax_L.set_xlabel("Time (Frame)", fontsize=14)
    ax_L.set_yticks([]) 
    ax_L.grid(True, linestyle=':', alpha=0.6)
    
    line1, = ax_L.plot(state['frame_history'], state['Lf_history'], 'r-', linewidth=3, alpha=0.6)
    line2, = ax_L.plot(state['frame_history'], state['Lb_history'], 'b-', linewidth=3, alpha=0.6)
    line3, = ax_L.plot(state['frame_history'], state['L_history'], 'g-', linewidth=4)
    
    ax_L.legend([line1, line2, line3], 
                ['Upper Body L', 'Lower Body L', 'Total L'], 
                loc='upper right', frameon=True, fontsize=12)
    
    if physics_explanation:
        props = dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='gray')
        ax_L.text(0.5, 0.25, physics_explanation, transform=ax_L.transAxes, 
                  ha='center', va='center', fontsize=16, color='#222', bbox=props)

# --- 运行 ---

fig = plt.figure(figsize=(12, 14))
# 使用 GridSpec 定义布局
gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1]) 
ax_3d = fig.add_subplot(gs[0], projection='3d')
ax_L = fig.add_subplot(gs[1])

total_frames = frames_bend + frames_twist1 + frames_twist2 + frames_unbend + frames_end + frames_wait1 + frames_wait2 + frames_wait3

# 增加 fps 以保证平滑
anim = FuncAnimation(fig, update, frames=total_frames, interval=30, repeat=False)

# 保存为 GIF (如果需要保存，取消下面两行的注释)
print("Generating Animation...")
# anim.save("cat_reflex_final.gif", writer=PillowWriter(fps=30))
print("Done.")
anim.save("cat_reflex.gif", writer=PillowWriter(fps=30))
plt.tight_layout()
plt.show()