from pathlib import Path
from huggingface_hub import snapshot_download
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

def main():
    dataset_root = Path("./datasets/MVTecAD")
    carpet_dir = dataset_root / "carpet"

    # 1. Automatically download only the 'carpet' folders from Hugging Face if missing
    if not (carpet_dir / "train").exists():
        print("Downloading MVTec 'carpet' dataset from Hugging Face (~250MB)...")
        snapshot_download(
            repo_id="aristrops/mvtec-concepts",
            repo_type="dataset",
            local_dir=str(dataset_root),
            allow_patterns=[
                "carpet/train/**",
                "carpet/test/**",
                "carpet/ground_truth/**",
            ],
        )
        print("Dataset download complete!")

    print("Loading local MVTec AD dataset and starting training...")
    datamodule = MVTecAD(
        root=str(dataset_root),
        category="carpet",
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
    )

    model = Patchcore(backbone="resnet18", pre_trained=True)

    engine = Engine(default_root_dir="./results")
    engine.fit(model=model, datamodule=datamodule)

    test_results = engine.test(model=model, datamodule=datamodule)
    print("\nFinal Test Results:")
    print(test_results)

if __name__ == "__main__":
    main()