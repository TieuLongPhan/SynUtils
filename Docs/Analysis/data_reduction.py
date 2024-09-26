import sys
import numpy as np
from pathlib import Path


def main():
    root_dir = Path(__file__).parents[2]
    sys.path.append(str(root_dir))

    from synutils.utils import load_compressed, save_compressed
    from synutils.Visualization.embedding import Embedding

    data = load_compressed(f"{root_dir}/Data/schneider50k_fp.npz")
    X = data[:, 1:]
    y = data[:, :1]

    emb = Embedding(cache_dir=f"{root_dir}/Data/Split/cachedir")
    X_ebd = emb.compute_tsne(X, cache=True)

    new_data = np.concatenate((y, X_ebd), axis=1)

    save_compressed(new_data, f"{root_dir}/Data/Split/data_embedding.npz")


if __name__ == "__main__":
    main()
