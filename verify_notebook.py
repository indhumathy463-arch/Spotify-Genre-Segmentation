import nbformat
import sys

with open('notebooks/spotify_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

error_count = 0
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                print(f"Error in cell {idx}: {output.ename} - {output.evalue}")
                error_count += 1

code_count = sum(1 for c in nb.cells if c.cell_type == 'code')
md_count = sum(1 for c in nb.cells if c.cell_type == 'markdown')

print(f"Total code cells: {code_count}")
print(f"Total markdown cells: {md_count}")
print(f"Execution Errors: {error_count}")

if error_count > 0:
    sys.exit(1)
print("Notebook passed 100% cleanly!")

