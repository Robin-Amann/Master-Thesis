from src.config_types.benchmark import DatasetSourceConfig
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download

def load_hf_dataset(config: DatasetSourceConfig, limit: int | None = None) -> Dataset:
    """Load a configured HF dataset; verify access and schema first."""
    
    kwargs = {
        "path": config.repo,
        "split": config.split,
    }
    if config.config is not None :
        kwargs["name"] = config.config

    ds: Dataset = load_dataset(**kwargs)

    if limit is not None and limit > 0 :
        ds = ds.select(range(min(limit, len(ds))))

    return ds


def group_by_media(dataset: Dataset) :
    return dataset


def download_hf_dataset_files(repo: str, revision: str | None = None, patterns: str | list[str] | None = None) : 
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,  # Can also be a commit hash
        # local_dir="./dataset",
        allow_patterns=patterns,
    )




# import os
# import zipfile
# from pathlib import Path

# download_hf_dataset_files("MME-Benchmarks/Video-MME-v2", patterns=[f"videos/{zip_id}.zip" for zip_id in zips])

# import pandas as pd

# # Assuming your Hugging Face dataset is loaded as a DataFrame
# dataset = load_hf_dataset(config.datasets[3].source).filter(lambda x: p(x), batched=False)
# df = dataset.to_pandas()

# counts = (
#     df.groupby(["group_type", "second_head", "third_head"], dropna=False)
#       .size()
#       .reset_index(name="count")
#       .sort_values(["group_type", "second_head", "third_head"])
# )

# # Print hierarchical output
# for group_type, group_df in counts.groupby("group_type", sort=True):
#     print(f"\n{group_type}")

#     for second_head, second_df in group_df.groupby("second_head", sort=True, dropna=False):
#         print(f"  - {second_head}")

#         for _, row in second_df.iterrows():
#             print(f"      - {row['third_head']}: {row['count']}")


# x = [ f"{(int((int(x)-1)/20) + 1):03d}" for x in x ]
# print(sorted(list(set(x))))
# print(dataset)
# video_ids = sorted(list(set(dataset["video_id"])))
# zip_ids = set([ int(int(x)/20) for x in video_ids ])
# zip_ids = sorted(list(zip_ids))
# print(video_ids)
# print(zip_ids)

# def extract_and_filter(src_dir: Path, file_names: list[str], dest_dir: Path):
#     print(src_dir.is_dir())
#     dest_dir.mkdir(parents=True, exist_ok=True)
#     allowed_files = set(file_names)

#     # Extract ZIP archives
#     for zip_path in src_dir.glob("*.zip"):
#         print(zip_path.name)

#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(dest_dir)

#         # Delete files not in the provided list
#         for path in dest_dir.rglob("*"):
#             if path.is_file():
#                 if path.stem not in allowed_files:
#                     path.unlink() # remove the file


# zips = [
#     '003', '004', '006', '008', 
#     '009', '010', '011', '014', 
#     '018', '019', '021', '023', 
#     '024', '026', '027', '029', 
#     '030', '032', '034', '036', 
#     '038', '039', '040'
# ]
# files = ['060', '079', '111', '112', '113', '158', '159', '160', '161', '162', '182', '196', '197', '215', '216', '217', '218', '219', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '358', '359', '360', '361', '362', '407', '408', '409', '443', '444', '476', '477', '478', '479', '480', '506', '518', '519', '520', '521', '522', '528', '561', '562', '594', '595', '596', '597', '598', '636', '637', '638', '639', '640', '680', '712', '713', '714', '715', '758', '759', '760','761', '762', '784']

# extract_and_filter(
#     src_dir=Path("/export/data1/ramann/huggingface/hub/datasets--MME-Benchmarks--Video-MME-v2/snapshots/6e4bebb03202e1ddbf3d37703e560e51c5aa2d64/videos/"),
#     file_names=files,
#     dest_dir=Path("/export/data1/ramann/huggingface/hub/Video-MME-v2")
# )
