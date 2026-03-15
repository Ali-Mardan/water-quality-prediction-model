import json
with open('Benchmark_Model_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('out2.txt', 'w', encoding='utf-8') as f2:
    for c in nb['cells']:
        if c['cell_type'] == 'code' and 'train_test_split' in ''.join(c['source']):
            f2.write(''.join(c['source']) + '\n-----\n')
