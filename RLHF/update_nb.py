import json
with open("result.ipynb", "r") as f:
    nb = json.load(f)
for cell in nb["cells"]:
    if cell.get("id") == "cell-4":
        source = "".join(cell["source"])
