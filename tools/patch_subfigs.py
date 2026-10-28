import re
p='main.tex'; s=open(p,encoding='utf-8').read()
if '\\usepackage{subcaption}' not in s:
    s=s.replace('\\usepackage{placeins}','\\usepackage{placeins}\n\\usepackage{subcaption}',1)
CFG={
 'fig01_experimental_curves':(6,3,0.32,r" Panels (a)--(f) correspond to \ns{} = 0, 1, 2, 3, 4 and 5\%, respectively."),
 'p3fig02_modes':(2,2,0.48,""),
 'fig04_predicted_ns_curves':(3,1,0.98,r" Rows (a)--(c) show the gradient-boosted tree, the deep network and the physics-informed network, respectively."),
 'fig06_blind_curves':(5,3,0.32,r" Panels (a)--(e) correspond to the five withheld mixtures $(1.0,0.5)$, $(2.0,1.5)$, $(3.0,2.0)$, $(4.0,1.0)$ and $(5.0,1.5)$ in (\ns, \pva)."),
 'fig07_parity_blind':(4,1,0.98,r" Panels (a)--(d) present the sixteen formulations in four groups of four."),
 'fig09_ablation':(4,2,0.48,r" Panels (a)--(d) show the four ablation arms."),
 'p2fig04_smoothness':(2,2,0.48,""),
 'fig10_digital_twin':(2,2,0.48,""),
 'fig11_uncertainty':(5,3,0.32,r" Panels (a)--(e) correspond to the five withheld mixtures."),
 'fig12_response_surface':(4,2,0.48,r" Panels (a)--(d) correspond to \pva{} = 0.5, 1.0, 1.5 and 2.0\%, respectively."),
 'fig13_reviewer_analytics':(2,2,0.48,""),
}
L='abcdefghijklmnopqrstuvwxyz'
RX_INC=r'\\resizebox\{[^}]*\}\{[^}]*\}\{\\input\{figures/%s\.pgf\}\}'
tot=0; report=[]
blocks=list(re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}', s, re.S))
out=[]; last=0
for m in blocks:
    blk=m.group(0)
    inc=re.search(r'\\input\{figures/([^}]+)\.pgf\}', blk)
    name=inc.group(1) if inc else None
    if name in CFG:
        n,cols,w,extra=CFG[name]
        body=''
        for i in range(n):
            body+=('\\begin{subfigure}{%.2f\\linewidth}\\centering\n'
                   '\\resizebox{\\linewidth}{!}{\\input{figures/%s_part%s.pgf}}\n'
                   '\\caption{}\n\\end{subfigure}'%(w,name,L[i]))
            body+= '\n\\par\\medskip\n' if (i+1)%cols==0 else '\\hfill\n'
        blk,k=re.subn(RX_INC%re.escape(name), lambda mm: body, blk, count=1)
        if k!=1:
            report.append('RESIZEBOX-NOMATCH '+name)
        else:
            tot+=1
        if extra:
            blk,k2=re.subn(r'(\\caption\{)(.*?)(\}\s*\n\s*\\label\{)',
                           lambda mm: mm.group(1)+mm.group(2)+extra+mm.group(3), blk, count=1, flags=re.S)
            if k2!=1: report.append('CAPTION-NOMATCH '+name)
    out.append(s[last:m.start()]); out.append(blk); last=m.end()
out.append(s[last:])
open(p,'w',encoding='utf-8').write(''.join(out))
print('figures converted to subfigures:', tot, 'of', len(CFG))
print('problems:', report if report else 'none')
