import kagglehub
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
raw_data = project_root / "data" / "raw"

path = kagglehub.dataset_download(
    'asaniczka/tmdb-movies-dataset-2023-930k-movies',
    output_dir=raw_data)

print('Path to dataset files:', path)
