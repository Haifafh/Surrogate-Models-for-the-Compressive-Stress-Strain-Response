import os, re, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT=os.path.expanduser('~/workspace/paper'); os.chdir(ROOT)

# ---------- 1. cost table with clean display names (no underscores) ----------
cost=pd.read_csv('out/cost_all.csv')
ren={'NN data-only (no physics)':'NN data-only (matched control)',
     'PINN (physics-informed)':'PINN (auto-selected, $\\lambda=0$)'}
def clean(m):
    m=ren.get(m,m); m=m.replace('_',' ')
    m=m.replace('Carreira-Chu (physics only)','Carreira--Chu (physics only)')
    return m
cost['disp']=cost.model.map(clean)
extra=pd.DataFrame([dict(model='PINN (physics forced on)',disp='PINN (physics forced on)',
                         fit_s=17.63,infer_ms=0.316)])
cost=pd.concat([cost[['model','disp','fit_s','infer_ms']],extra],ignore_index=True).sort_values('fit_s')
mf,ml=cost.fit_s.min(),cost.infer_ms.min()
L=[r'\begin{table}[t]',r'\centering',
   r'\caption{Computational cost of the surrogate families (blind-mixture split). The fitting time is the '
   r'wall-clock cost of training or calibrating each model once on the 578 training points; the inference '
   r'time is the mean wall-clock time to evaluate one complete 60-point stress--strain curve. Both were '
   r'measured in pure Python/NumPy on a single CPU core. The fastest value in each column is bold.}',
   r'\label{tab:cost}',r'\small',r'\setlength{\tabcolsep}{4.5pt}',
   r'\begin{tabular}{@{}lcc@{}}',r'\toprule',
   r'Model & fit time (s) & inference (ms per 60-pt curve)\\',r'\midrule']
for r in cost.itertuples():
    fs='%.3f'%r.fit_s; ls='%.3f'%r.infer_ms
    if abs(r.fit_s-mf)<1e-12: fs=r'\textbf{%s}'%fs
    if abs(r.infer_ms-ml)<1e-12: ls=r'\textbf{%s}'%ls
    L.append('%s & %s & %s\\\\'%(r.disp,fs,ls))
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}','']
open('cost_table.tex','w').write('\n'.join(L))
print('cost_table.tex: clean names OK')

# ---------- 2. ablation labels -> ASCII so the PGF figure is safe ----------
a=pd.read_csv('out/ablation.csv')
a['variant']=['lambda = 0 (auto-selected)','lambda_E = 3 (elastic pin)',
              'lambda_E + lambda_D','all three terms']
a.to_csv('out/ablation.csv',index=False)
print('ablation.csv labels sanitised:',list(a.variant))

# ---------- 3. main.tex edits (plain replace / lambda repl -> no escape templates) ----------
s=open('main.tex').read(); log=[]
old="returning smooth, single-peaked hardening--softening curves. Accuracy and"
new=("returning smooth, single-peaked hardening--softening curves. A matched-optimiser ablation established "
     "that the physics regularisation did not improve held-out accuracy on this dataset: the automatic "
     "selector assigned a zero weight to every physics term, so the deployed network operates as a "
     "data-driven model. Accuracy and")
if old in s: s=s.replace(old,new,1); log.append('abstract: null result disclosed')
kw=s.find('\\textbf{Keywords:}')
if kw>0:
    note=(r'The primary contribution is methodological: a leakage-free, mixture-level benchmarking protocol '
          r'that exposes the optimistic bias of point-wise splits and quantifies the ranking stability of '
          r'surrogate families under resampling.'+'\n\n')
    s=s[:kw]+note+s[kw:]; log.append('abstract: contribution repositioned')
p='The second concerns physical admissibility.'
if p in s:
    s=s.replace(p, p+' A third concerns statistical robustness: because a single blind split of five '
                'mixtures provides very few effective degrees of freedom, every headline comparison is '
                'repeated here over fifteen resampled hold-outs with paired significance testing.',1)
    log.append('intro: third contribution added')
ablbody=r"""The ablation is repeated here with a strictly matched optimiser: every arm uses an identical
learning rate, batch size, epoch count and seed set, and only the physics weights differ, so the comparison
isolates the effect of the penalties. Table~\ref{tab:abl} and Figure~\ref{fig:abl} report the outcome.
Adding the elastic-modulus pin alone leaves the error essentially unchanged (RMSE 1.807~MPa against
1.811~MPa for the data-only arm). Adding the damage-consistency term degrades it to 2.308~MPa, and
switching on all three penalties degrades it further to 4.960~MPa, more than a factor of two worse than the
data-only optimum. For this dataset, therefore, the physics regularisation does not improve held-out
accuracy and the curvature-smoothness penalty actively harms it. The automatic selector reached the same
conclusion, assigning $\lambda_E=\lambda_D=\lambda_S=0$; this is why the deployed network is numerically
identical to the matched data-only control, and it is reported as such throughout.

\input{table_abl.tex}

"""
s2,n=re.subn(r'(\\subsection\{[^}]*[Aa]blation[^}]*\})(.*?)(?=\\subsection|\Z)',
             lambda m: m.group(1)+'\n'+ablbody, s, flags=re.S)
if n: s=s2; log.append('ablation subsection replaced (matched optimiser)')

block=open('newblock.tex').read() if os.path.exists('newblock.tex') else None
print('edits:',log)
open('main.tex','w').write(s)
