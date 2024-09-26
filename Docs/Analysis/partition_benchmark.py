import sys
import pandas as pd
from pathlib import Path


def main():
    root_dir = Path(__file__).parents[2]
    sys.path.append(str(root_dir))
    from synutils.Analysis.split_benchmark import SplitBenchmark
    from synutils.utils import load_compressed, save_dict_to_json

    data = load_compressed(f"{root_dir}/Data/schneider50k_fp.npz")
    data = pd.DataFrame(data)
    data.rename(columns={0: "class_id"}, inplace=True)
    benchmark = SplitBenchmark(
        data=data,
        test_size=0.2,
        class_columns="class_id",
        random_state=42,
        drop_class_ratio=0.1,
    )
    results = benchmark.fit(number_sample=5, n_jobs=3, verbose=2)
    save_dict_to_json(results, f"{root_dir}/Data/Split/benchmark_results.json")


if __name__ == "__main__":
    main()
