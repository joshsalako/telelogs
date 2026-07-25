import json

path = "/Users/joshua/Downloads/telelogs/grpo.ipynb"
with open(path) as f:
    nb = json.load(f)

patch_code = r"""
# --- INSPECT QWEN3.5 COMPUTE_3D_POSITION_IDS ---
import transformers

hf_file = transformers.models.qwen3_5.modeling_qwen3_5.__file__
with open(hf_file, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "def compute_3d_position_ids" in line:
        print(f"Found compute_3d_position_ids at line {idx+1}:")
        for j in range(idx, min(len(lines), idx + 35)):
            print(f"{j+1}: {lines[j]}", end="")
        break
# ---------------------------------------
"""

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "INSPECT" in "".join(cell.get("source", [])) or "PERMANENT FIX" in "".join(cell.get("source", [])):
        cell["source"] = [line + "\n" for line in patch_code.strip().split("\n")]
        break

with open(path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Updated cell to inspect compute_3d_position_ids.")
