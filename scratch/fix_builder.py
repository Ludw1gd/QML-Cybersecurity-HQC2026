with open('scratch/build_full_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace cells.append(nbf.v4.new_code_cell(""" with r"""
content = content.replace('cells.append(nbf.v4.new_code_cell("""', 'cells.append(nbf.v4.new_code_cell(r"""')

# Remove escaped quotes in docstrings
content = content.replace('\\"\\"\\"Mengambil sampel seimbang per kelas dengan pengacakan terkontrol.\\"\\"\\"', '# Mengambil sampel seimbang per kelas dengan pengacakan terkontrol')

# In cell 15, replace \\n with \n in raw string if any
content = content.replace(r'print(f"\\nMenjalankan {exp_name}...")', r'print(f"\nMenjalankan {exp_name}...")')
content = content.replace(r'print("\n=== TABEL ABLATION STUDY: MULTI-SEED BENCHMARK (MEAN ± STD, 5 SEEDS) ===")', r'print("\n=== TABEL ABLATION STUDY: MULTI-SEED BENCHMARK (MEAN ± STD, 5 SEEDS) ===")')
content = content.replace(r'print("\n=== HASIL TEMPORAL SPLIT (SUBSET SEIMBANG) ===")', r'print("\n=== HASIL TEMPORAL SPLIT (SUBSET SEIMBANG) ===")')
content = content.replace(r'print(f"\n=== EVALUASI PADA SELURUH 46.769 BARIS TRAFIK MASA DEPAN (UNBALANCED) ===")', r'print(f"\n=== EVALUASI PADA SELURUH 46.769 BARIS TRAFIK MASA DEPAN (UNBALANCED) ===")')

with open('scratch/build_full_notebook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('build_full_notebook.py updated successfully')
