import json
with open('Benchmark_Model_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for c in nb['cells']:
    if c['cell_type'] == 'code' and 'train_test_split' in ''.join(c['source']):
        print(''.join(c['source']))
        print('-----')
