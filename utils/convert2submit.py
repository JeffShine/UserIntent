import json
from pathlib import Path
import pandas as pd


def convert2submit(test_file: Path, prediction_file: Path, save_path: Path):
    save_data = []
    for line in open(prediction_file, "r"):
        prediction_data = json.loads(line)
        save_data.append(prediction_data)

    df = pd.DataFrame(save_data)
    df.to_csv(save_path, index=None, encoding="utf-8-sig")


if __name__ == "__main__":
    test_file = "/home/pubw/proj/LLaMA-Factory/data/mire/test1_modified.json"
    prediction_file = "/home/pubw/proj/LLaMA-Factory/OurTry/save/cot/12021225/output.jsonl"
    save_path = "submit_cot.csv"
    convert2submit(test_file, prediction_file, save_path)

# end main
