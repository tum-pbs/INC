import numpy as np
import matplotlib.pyplot as plt
def create_2d_roll(ax, data, title, extent, cmap="RdBu"):
    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower", extent=extent)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
    return im

def create_1d_roll(ax, pred, target, x, times, dt):
    colors = plt.cm.RdBu(np.linspace(0, 1, len(times)))
    for idx, t in enumerate(times):
        time_val = t * dt
        ax.plot(x, pred[t], color=colors[idx], label=f"pred t={time_val:.4f}s")
        ax.scatter(x, target[t], color=colors[idx], marker='o', s=5, label=f"target t={time_val:.4f}s" if idx == 0 else "")
        # ax.scatter(x, target[t], color=colors[idx], marker='o', s=5, label=f"target t={time_val:.4f}s")

    ax.set_title("1D Rollout Comparison")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u(x,t)")
    ax.legend(loc="best", fontsize=8)

def plot_result(pred_data, target_data, train_params, mode, batch=0):
    pred_rollout = pred_data[batch]
    target_rollout = target_data[batch]
    T, D = pred_rollout.shape
    # x = np.linspace(0, train_params.metadata["L"], D)
    x = np.linspace(0, D, D)
    # extent = (0, train_params.metadata["L"], 0, T * train_params.dt) 
    extent = (0, D, 0, T * train_params.dt)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    create_2d_roll(ax[0], pred_rollout, f"{mode} Predicted", extent)
    create_2d_roll(ax[1], target_rollout, f"{mode} Target", extent)
    
    # desired_times = np.arange(0, T * train_params.dt + 0.1, 0.2)
    desired_times = np.linspace(0, T * train_params.dt, num=10)
    times = np.clip((desired_times / train_params.dt).astype(int), 0, T-1)
    create_1d_roll(ax[2], pred_rollout, target_rollout, x, times, train_params.dt)
    
    fig.tight_layout()
    return fig

def plot_correction(correction_data, u, u_target,train_params, batch=0):
    """Plot correction term spatial profiles at selected time steps"""
    correction = correction_data[batch]  # [T, D]
    u = u[batch]  # [T, D]
    u_target = u_target[batch]  # [T, D]
    T, D = correction.shape
    x = np.linspace(0, D, D)

    # x = np.linspace(0, train_params.metadata["L"], D)
    
    # Sample time steps
    desired_times = np.linspace(0, T * train_params.dt, num=10)
    times = np.clip((desired_times / train_params.dt).astype(int), 0, T-1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    
    for idx, t in enumerate(times):
        ax[0].plot(x, correction[t], color=colors[idx], label=f"t={t*train_params.dt:.4f}s ")
        ax[1].plot(x, u[t], color=colors[idx], label=f"t={t*train_params.dt:.4f}s ")
        ax[1].scatter(x, u_target[t], color=colors[idx], marker='o', s=5, label=f"target t={t*train_params.dt:.4f}s " if idx == 0 else "")
    # ax[0].set_title("Correction Term Spatial Profile")
    ax[0].set_ylabel("Correction Value")
    ax[1].set_ylabel("u(x,t)")
    for axs in [ax[0], ax[1]]:
        axs.set_xlabel("x (m)")
        axs.legend()
        axs.grid(True)
    return fig
