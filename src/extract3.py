import json
with open('Benchmark_Model_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('out3.txt', 'w', encoding='utf-8') as f2:
    for c in nb['cells']:
        if c['cell_type'] == 'code' and ('split_data' in ''.join(c['source'])):
            f2.write(''.join(c['source']) + '\n-----\n')
