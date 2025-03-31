import hydra
import torch
import numpy as np

from omegaconf import DictConfig

from torch_brain.registry import MODALITY_REGISTRY
from finetune import load_model_from_ckpt

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

import ipdb



theme_blue_light   = sns.color_palette('Blues')[0]
theme_blue_mid     = sns.color_palette('Blues')[2]
theme_blue_dark    = sns.color_palette('Blues')[3]
theme_orange_light = sns.color_palette('Oranges')[0]
theme_orange_mid   = sns.color_palette('Oranges')[2]
theme_orange_dark  = sns.color_palette('Oranges')[3]
theme_coral_light  = (255/255, 204/255, 204/255)
theme_coral_mid    = (247/255, 156/255, 156/255)
theme_coral_dark   = (232/255, 107/255, 107/255)
theme_green_light  = (206/255, 242/255, 238/255) 
theme_green_mid    = (120/255, 204/255, 194/255)
theme_green_dark   = (74/255,  189/255, 175/255)
 
corals  = [theme_coral_light,  theme_coral_mid,  theme_coral_dark]
greens  = [theme_green_light,  theme_green_mid,  theme_green_dark]
blues   = [theme_blue_light,   theme_blue_mid,   theme_blue_dark]
oranges = [theme_orange_light, theme_orange_mid, theme_orange_dark]



def visualize_assist_emb(assist_emb):

    weights = assist_emb.weight.data.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(
        weights.T,
        cbar=True,
        # xticklabels=10,
        # yticklabels=10,
        ax=ax,
    )

    assist_labels = [
        '0 to 0.1',
        '0.1 to 0.3',
        '0.3 to 0.5',
        '0.5 to 0.7',
        '0.7 to 1',
        '-1']
    
    ax.invert_yaxis()
    ax.set_xticklabels(assist_labels, rotation=30, ha='right')   
    ax.set_xlabel('Assist Value Bins')
    ax.set_ylabel('Assist Embedding Dimension')
    plt.show()


def visualize_unit_emb(
    unit_emb, 
    ckpt_id,
    dim_reduction_method='umap',
    color_by='array'):

    weights = unit_emb.weight.data.cpu().numpy()
    vocab = unit_emb.vocab

    unit_ids_by_array = {
        'MC-LAT': [],
        'MC-MED': [],
        'PPC-SPL': [],
        'PPC-IPL': [],
    }

    unit_ids_by_session = {}

    for vocab_key in vocab.keys():

        if vocab_key == 'NA':
            continue

        unit_id = vocab[vocab_key]

        vocab_key_ = vocab_key.split('/')
        session_string = vocab_key_[1]
        array_string = vocab_key_[2]
        array = array_string[6:]

        unit_ids_by_array[array].append(unit_id)

        if session_string not in unit_ids_by_session:
            unit_ids_by_session[session_string] = []

        unit_ids_by_session[session_string].append(unit_id)


    if dim_reduction_method == 'pca':
        reducer = PCA(n_components=2, random_state=0)
        weights_ = reducer.fit_transform(weights)
        plot_title = "Unit embeddings (PCA to 2D), colored by array"

    elif dim_reduction_method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=0, verbose=1)
        weights_ = reducer.fit_transform(weights)
        plot_title = "Unit embeddings (t-SNE to 2D), colored by array"

    elif dim_reduction_method == 'umap':
        # Common hyperparameters: n_neighbors=15, min_dist=0.1, etc.
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        weights_ = reducer.fit_transform(weights)
        plot_title = "Unit embeddings (UMAP to 2D)"


    if color_by == 'array':
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        colors = [
            theme_blue_mid,
            theme_blue_dark,
            theme_orange_mid,
            theme_orange_dark,
        ]

        # ---- Subplot 1: color-coded by array ----
        for i, (array_name, unit_indices) in enumerate(unit_ids_by_array.items()):
            coords = weights_[unit_indices]  # shape => (len(unit_indices), 2)
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                label=array_name,
                alpha=0.5,
                s=10,
                color=colors[i]
            )
        ax.legend()
        ax.set_title(f"Unit embeddings ({dim_reduction_method} to 2D), colored by array")
        ax.set_xlabel("Dim1")
        ax.set_ylabel("Dim2")
    
    elif color_by == 'session':
        for i, (session_name, unit_indices) in enumerate(unit_ids_by_session.items()):

            if i > 10:
                break

            coords = weights_[unit_indices]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                label=session_name,
                alpha=0.5,
                s=10
            )
        ax.legend()
        ax.set_title(f"Unit embeddings ({dim_reduction_method} to 2D), colored by session")
        ax.set_xlabel("Dim1")
        ax.set_ylabel("Dim2")

    plt.show()
    fig.savefig(f"{ckpt_id}_unit_embeddings_{dim_reduction_method}_{color_by}.png", dpi=300, transparent=True)

    plt.close(fig)


def visualize_unit_emb_animate(unit_emb, ckpt_id, dim_reduction_method='pca'):
    """
    1) Gather all unit embeddings (N x 64).
    2) Reduce to 2D using PCA, t-SNE, or UMAP.
    3) Create an animation where each frame highlights the units in one session,
       colored by array, with all other units in gray background.
    4) Show a legend indicating which color corresponds to each array.
       
    Args:
        unit_emb: your embedding module, having:
            - .weight: (num_units, embed_dim=64)
            - .vocab: dict { "unit_string": index }
        dim_reduction_method: str, one of ['pca', 'tsne', 'umap']
        
    Returns:
        ani: A matplotlib.animation.FuncAnimation object
    """

    # --------------------------
    # 0) Retrieve weights & vocab
    # --------------------------
    all_weights = unit_emb.weight.data.cpu().numpy()  # shape (num_units, 64)
    vocab = unit_emb.vocab  # dict: { "unit_str": index }

    # --------------------------
    # 1) Build groupings by session & array
    # --------------------------
    session_to_indices = {}
    array_for_unit = [None] * len(all_weights)

    for unit_str, unit_idx in vocab.items():
        if unit_str == 'NA':
            continue

        # Example: "ALMG103SSCO_OT_OL/n2_20240222_CenterOut/group_MC-MED/elec0/unit_0"
        parts = unit_str.split('/')
        session_str = parts[1]         # e.g. "n2_20240222_CenterOut"
        array_str = parts[2][6:]       # remove "group_" => e.g. "MC-MED"

        # group by session
        if session_str not in session_to_indices:
            session_to_indices[session_str] = []
        session_to_indices[session_str].append(unit_idx)

        # remember each unit's array to color it appropriately
        array_for_unit[unit_idx] = array_str

    # We'll build a consistent color mapping for arrays
    unique_arrays = sorted(set(arr for arr in array_for_unit if arr is not None))
    # Some default matplotlib tab colors
    default_colors = [
        sns.color_palette('Blues')[2], 
        sns.color_palette('Blues')[4], 
        sns.color_palette('Oranges')[2],
        sns.color_palette('Oranges')[4],
    ]

    color_map = {}
    for i, arr_name in enumerate(unique_arrays):
        color_map[arr_name] = default_colors[i % len(default_colors)]

    # --------------------------
    # 2) Dimensionality reduction
    # --------------------------
    if dim_reduction_method == 'pca':
        reducer = PCA(n_components=2, random_state=0)
        coords_2d = reducer.fit_transform(all_weights)
        method_str = "PCA"
    elif dim_reduction_method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=0, verbose=1)
        coords_2d = reducer.fit_transform(all_weights)
        method_str = "t-SNE"
    elif dim_reduction_method == 'umap':
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0)
        coords_2d = reducer.fit_transform(all_weights)
        method_str = "UMAP"
    else:
        raise ValueError("method must be 'pca', 'tsne', or 'umap'.")

    # For convenience:
    all_x = coords_2d[:, 0]
    all_y = coords_2d[:, 1]

    # consistent axis limits across frames
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    # --------------------------
    # 3) Prepare the figure
    # --------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background scatter of *all* points in gray
    ax.scatter(all_x, all_y, c="lightgray", alpha=0.2, s=5, label="All units (background)")

    # We'll create another scatter for the current session:
    session_scat = ax.scatter([], [], s=20)

    # ax.set_xlim(x_min - 1, x_max + 1)
    # ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")

    # We'll set a "session" title later in update()
    ax.set_title(f"Unit embeddings ({method_str} to 2D)")

    # --- Create a legend for arrays ---
    # We'll build handles from the color_map:
    array_legend_handles = []
    for arr_name, clr in color_map.items():
        patch = mpatches.Patch(color=clr, label=arr_name)
        array_legend_handles.append(patch)
    # You can also add a patch for the background if you want:
    array_legend_handles.append(mpatches.Patch(color="lightgray", label="Units from other sessions"))
    ax.legend(handles=array_legend_handles, loc="best")

    # List out sessions in a stable, sorted order
    session_names = sorted(session_to_indices.keys())

    # --------------------------
    # 4) Define update func
    # --------------------------
    def update(frame_idx):
        """Highlight the units for the session at `frame_idx`."""
        session_name = session_names[frame_idx]
        unit_indices = session_to_indices[session_name]

        # subset of coords
        session_coords = coords_2d[unit_indices]
        xs = session_coords[:, 0]
        ys = session_coords[:, 1]

        # array-based colors
        colors = []
        for idx in unit_indices:
            arr_name = array_for_unit[idx]
            c = color_map.get(arr_name, "black")
            colors.append(c)

        # update the scatter
        session_scat.set_offsets(np.column_stack((xs, ys)))
        session_scat.set_facecolor(colors)
        session_scat.set_edgecolor("none")

        ax.set_title(f"Session: {session_name} "
                     f"({method_str} 2D) [{frame_idx+1}/{len(session_names)}]")

        return (session_scat,)

    # --------------------------
    # 5) Build the animation
    # --------------------------
    ani = animation.FuncAnimation(
        fig,
        update,                 # update function
        frames=len(session_names),
        interval=1000,          # ms between frames -> 1 second per frame
        blit=False,
        repeat=False
    )

    ani.save(f"{ckpt_id}_unit_embeddings_{dim_reduction_method}.gif", writer="imagemagick", fps=1)

    # plt.show()
    return ani


def visualize_session_emb():
    pass




@hydra.main(version_base="1.3", config_path="./configs", config_name="visualize_poyo_mp.yaml")
def main(cfg: DictConfig):

    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype("str")),
    ])

    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITY_REGISTRY)
    load_model_from_ckpt(model, cfg.ckpt_path)

    ckpt_id = cfg.ckpt_path.split("/")[-3]

    # visualize_assist_emb(model.assist_emb)
    visualize_unit_emb(model.unit_emb, ckpt_id, dim_reduction_method='pca', color_by='array')
    visualize_unit_emb_animate(model.unit_emb, ckpt_id, dim_reduction_method='umap')

    # ipdb.set_trace()


if __name__ == "__main__":

    main()

    

    