import os
import subprocess

os.makedirs("datasets", exist_ok=True)
print("downloading MultiWOZ 2.2...")
subprocess.run([
    "python", "-c",
    "from datasets import load_dataset; ds = load_dataset('pfb30/multi_woz_v22', trust_remote_code=True); "
    "import os; os.makedirs('datasets/multiwoz_data', exist_ok=True); "
    "[ds[s].to_json(f'datasets/multiwoz_data/multiwoz_{s}.json') for s in ['train', 'validation', 'test']]"
])
print("download complete! Large audio files MUSAN please download from PACE-ICE /scratch: https://www.openslr.org/resources/17/musan.tar.gz")