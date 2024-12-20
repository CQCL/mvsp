import os

api_ref_dir = "docs/source/api_reference"  # Update with the actual path if different
index_file = (
    "docs/source/api_reference/index.rst"  # Update with the actual path if different
)

# Fetch all .rst files in api_reference directory
rst_files = [
    f
    for f in os.listdir(api_ref_dir)
    if os.path.isfile(os.path.join(api_ref_dir, f)) and f.endswith(".rst")
]

# Exclude the index.rst file if it exists in the api_reference directory
rst_files = [f for f in rst_files if f != "index.rst"]

# Create the toctree entry for each file
toctree_entries = [
    # f"   {api_ref_dir}/{os.path.splitext(f)[0]}" for f in sorted(rst_files)
    f"   {os.path.splitext(f)[0]}"
    for f in sorted(rst_files)
]

# Create the toctree block
toctree_block = (
    ".. toctree::\n"
    "   :maxdepth: 2\n"
    "   :caption: API Reference:\n\n" + "\n".join(toctree_entries)
)

# Append or update the toctree block in index.rst
# Note: You might want to adjust this to insert the toctree block at a specific position
with open(index_file, "a") as index:
    index.write(toctree_block)

print(f"Updated {index_file} with {len(rst_files)} API reference entries.")
