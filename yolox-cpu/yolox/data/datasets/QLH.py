import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# ========== 心形参数方程 ==========
def heart_surface(u, v, scale=1.0, stretch=1.0):
    """
    生成心形曲面的坐标
    u: 纵向角度 (0 到 pi)
    v: 横向角度 (0 到 2pi)
    scale: 整体缩放因子 (心跳效果)
    stretch: 纵向拉伸因子 (让心跳更生动)
    """
    # 基础形状
    x0 = 16 * np.sin(u) ** 3 * np.cos(v)
    y0 = 16 * np.sin(u) ** 3 * np.sin(v)
    z0 = 13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)

    # 动态变形: 缩放整体，并稍微拉伸 z 轴模拟心跳
    x = x0 * scale
    y = y0 * scale
    z = z0 * scale * stretch  # 纵向额外变化，更有“跳动感”
    return x, y, z


# 生成网格 (高分辨率让跳动更平滑)
u = np.linspace(0, np.pi, 120)
v = np.linspace(0, 2 * np.pi, 120)
u, v = np.meshgrid(u, v)

# 预先计算基础坐标 (不跳动时的原始形状)
x0, y0, z0 = heart_surface(u, v, scale=1.0, stretch=1.0)

# ========== 创建图形 ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 初始曲面 (占位，动画中会实时更新)
surf = ax.plot_surface(x0, y0, z0,
                       cmap='Blues',
                       edgecolor='none',
                       alpha=0.95,
                       antialiased=True)

# 隐藏坐标轴和网格
ax.axis('off')
ax.grid(False)
ax.set_facecolor('white')

# 固定坐标轴范围 (留出跳动空间)
max_range = 20
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

# 设置一个好看的视角
ax.view_init(elev=30, azim=45)


# ========== 心跳动画 ==========
def animate(frame):
    # 心跳周期: 用正弦函数模拟，但心跳是快速膨胀、缓慢恢复
    # 使用 abs(sin(...)) 或 自定义波形来实现“怦然心动”的感觉
    # 这里用 mod 180 帧一个周期: 前30帧快速扩大，后150帧缓慢收缩
    t = frame % 180
    if t < 30:
        # 快速膨胀阶段 (模拟收缩期)
        factor = 1 + (t / 30) * 0.15  # 最大放大到 1.15 倍
        stretch = 1 + (t / 30) * 0.08  # 纵向额外拉伸
    else:
        # 缓慢恢复阶段
        factor = 1.15 - ((t - 30) / 150) * 0.15
        stretch = 1.08 - ((t - 30) / 150) * 0.08

    # 重新计算当前帧的心形坐标
    x_new = x0 * factor
    y_new = y0 * factor
    z_new = z0 * factor * stretch

    # 更新曲面的顶点数据
    surf.set_verts([np.array([x_new, y_new, z_new]).transpose(1, 2, 0)])

    # 可选: 颜色随心跳轻微加深 (增加活力)
    # 根据 factor 调整颜色映射的亮度 (跳动到最大时颜色更亮)
    color_brightness = 0.7 + factor * 0.3
    # 这里简单重设颜色 (使用 Blues 映射并调节亮度需要更复杂处理，为保持流畅先省略)
    # 但可以通过修改 facecolors 实现，但 set_verts 后自动重算颜色? 不会，需手动
    # 为了保证颜色也是动态的，我们手动重新设置 facecolors
    z_norm = (z_new - z_new.min()) / (z_new.max() - z_new.min())
    colors = plt.cm.Blues(z_norm * color_brightness)
    surf.set_facecolors(colors)

    return [surf]


ani = FuncAnimation(fig, animate, frames=360, interval=40, blit=False, repeat=True)

plt.title("💓 3D Beating Blue Heart | 怦然心动的蓝色爱心 💓", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 如果要保存为 GIF (需要安装pillow)
# ani.save('beating_blue_heart.gif', writer='pillow', fps=25)