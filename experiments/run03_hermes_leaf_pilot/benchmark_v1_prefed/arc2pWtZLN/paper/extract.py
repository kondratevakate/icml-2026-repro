import re
from bs4 import BeautifulSoup
html = open('/home/kate/projects/02_academia/icml-2026-repro/experiments/run03_hermes_leaf_pilot/arc2pWtZLN/paper/paper.html', encoding='utf-8', errors='ignore').read()
soup = BeautifulSoup(html, 'lxml')
for t in soup(['script','style']):
    t.extract()
text = soup.get_text('\n')
lines = [l.rstrip() for l in text.split('\n')]
text = '\n'.join(lines)
for kw in ['Definition 3.3','Proposition 3.1','Proposition 3.2','Theorem 3.2','Proposition 3.5','Corollary 3.6','Proposition 3.10','Algorithm 1','graded Laplacian','collapsed','S_{\\varepsilon}','S_','sigma_min','σ_min','protein','46.9','70.9','secondary structure','B_{k+1}','\\beta_{k+1}','gamma_k','B_{k+1}']:
    idxs = [m.start() for m in re.finditer(re.escape(kw), text)]
    print(f"{kw!r}: {len(idxs)} hits {idxs[:8]}")
print("\n=== total text length", len(text))
open('/home/kate/projects/02_academia/icml-2026-repro/experiments/run03_hermes_leaf_pilot/arc2pWtZLN/paper/paper.txt','w').write(text)
print("saved paper.txt")
