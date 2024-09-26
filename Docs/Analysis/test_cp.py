import pandas as pd
import numpy as np
from synutils.Partition.data_partition import DataPartition
from synutils.utils import setup_logging
from synutils.Uncertainty.peformance import cp_evaluate
from xgboost import XGBClassifier
from synutils.utils import save_dict_to_json

logger = setup_logging(log_filename="Data/cp_test.txt")


drop_ratio = np.arange(0.05, 1.0, 0.05)

total_results = {}
for ratio in drop_ratio:
    logger.info(f"DROP CLASS RATIO {ratio}")
    df = pd.read_csv("Data/schneider50k_fp.csv").iloc[:, 1:]
    split = DataPartition(
        df,
        test_size=0.2,
        class_column="class_id",
        method="stratified_class_reduction",
        random_state=42,
        drop_class_ratio=ratio,
        keep_data=True,
    )

    data_train, data_test, remove_train = split.fit()

    splitter_cablirated = DataPartition(
        data_train,
        test_size=0.3,
        class_column="class_id",
        method="stratified_target",
        random_state=42,
        drop_class_ratio=0.1,
        keep_data=True,
    )

    data_train, data_cal = splitter_cablirated.fit()

    logger.info("Train model...")
    # clf = load_model('Data/xgb.model')
    X_train = data_train.drop(["class_id", "class_mapping"], axis=1).values
    y_train = data_train["class_mapping"].values
    clf = XGBClassifier(random_state=42, n_jobs=3)
    clf.fit(X_train, y_train)

    logger.info("Evaluation...")
    results = cp_evaluate(
        clf, data_test, data_cal, "class_id", "class_mapping", remove_train, 0.05
    )
    total_results[ratio] = results
    logger.info(f"Results for {ratio}: {results}")

save_dict_to_json(total_results, "Data/results_cp_ratio.json")
