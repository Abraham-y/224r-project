"""Known-answer gate. Nothing launches until all FIVE gates (A-E) pass."""
import json,sys,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import surface_residual_probe as srp
import residual_reward as rr

# Repo root derived from this file's location, not hardcoded: the gate has to be
# runnable by anyone who checks the repo out, and a gate that only runs on one
# laptop is not a gate.
R=str(Path(__file__).resolve().parents[4])+"/"
P=R+"extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer"
X=np.load(P+".npz")["X"]; meta=json.load(open(P+".meta.json"))
oc=[json.loads(l) for l in open(R+"eval_c_outcome_n500.json") if l.strip()]
import re
# The REAL verifier, not a local regex+eval reimplementation. GATE D's reference
# numbers are a function of these labels, so the labels must come from the same
# code path as the RL reward. (The reimplementation this replaces agreed with it
# on all 7,775 answer-bearing C_outcome rollouts, so GATE D's targets are
# unchanged -- but agreement measured once is not a guarantee maintained.)
from evaluation.countdown import evaluate_equation, validate_equation
AN=re.compile(r"<answer>(.*?)</answer>",re.DOTALL)
def ok(eq,t,n):
    eq=eq.strip()
    if not validate_equation(eq,list(n)): return 0
    r=evaluate_equation(eq)
    return int(r is not None and abs(r-int(t))<1e-5)
texts,y,g=[],[],[]
for m in meta:
    r=oc[m["prompt_idx"]]; resp=r["response"][m["resp_idx"]]; mm=AN.search(resp)
    texts.append(resp); y.append(ok(mm.group(1),r["target"],r["nums"]) if mm else 0); g.append(m["prompt_idx"])
y=np.array(y); g=np.array(g)
print(f"rows={len(y)} prompts={len(set(g))} pos_rate={y.mean():.4f}")

# ---- GATE A: identity residualisation must equal a plain probe EXACTLY ----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import hashlib
te=np.array([int(hashlib.sha256(str(int(v)).encode()).hexdigest(),16)%2==0 for v in g]); tr=~te
plain=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=.05,class_weight="balanced")).fit(X[tr],y[tr])
keys=sorted(srp.surface_features(texts[0]))
ident=srp.SurfaceResidualProbe(keys,np.zeros(len(keys)),np.ones(len(keys)),
                               np.zeros((len(keys)+1,X.shape[1])),plain,identity=True)
a=plain.predict_proba(X[te])[:,1]; b=ident.predict_proba(X[te],text=[texts[i] for i in np.where(te)[0]])[:,1]
gA=np.array_equal(a,b)
print(f"\nGATE A  identity residual probe == plain probe : max|diff|={np.abs(a-b).max():.3e}  {'PASS' if gA else 'FAIL'}")

# ---- GATE B: a probe WITHOUT needs_text must get the original call signature ----
calls=[]
class Legacy:
    def predict_proba(self,Xa,*args,**kw):
        calls.append((Xa.shape,args,dict(kw))); return np.array([[0.3,0.7]])
rr._think_close_hidden=lambda *a,**k: np.zeros(X.shape[1],dtype=np.float32)
rr._reconstruct_prompt=lambda gt: "p"
fn=rr.make_probe_reward(None,None,Legacy(),16,"probe",0.0,lambda s,gt: 1.0)
v=fn("<think>x</think><answer>1+1</answer>",{"_prompt":"p"})
gB=(len(calls)==1 and calls[0][1]==() and calls[0][2]=={} and abs(v-0.7)<1e-9)
print(f"GATE B  legacy probe called as predict_proba(h[None,:]) only : calls={calls}  reward={v}  {'PASS' if gB else 'FAIL'}")

# ---- GATE C: text-aware probe actually receives the text ----
got=[]
class TA:
    needs_text=True
    def predict_proba(self,Xa,text=None): got.append(text); return np.array([[0.1,0.9]])
fn2=rr.make_probe_reward(None,None,TA(),16,"probe",0.0,lambda s,gt: 1.0)
s="<think>abc</think><answer>2+2</answer>"; v2=fn2(s,{"_prompt":"p"})
gC=(got==[s] and abs(v2-0.9)<1e-9)
print(f"GATE C  text-aware probe receives solution_str verbatim : {'PASS' if gC else 'FAIL'}")

# ---- GATE D: the real residualised probe reproduces the measured 0.836/0.978 ----
probe,rep=srp.fit(X,texts,y,g,C=.05)
gD=abs(rep["auroc_residual_heldout"]-0.8363)<0.02 and abs(rep["auroc_raw_heldout"]-0.9775)<0.02
print(f"GATE D  residual held-out AUROC={rep['auroc_residual_heldout']:.4f} (measured 0.8363) | "
      f"raw={rep['auroc_raw_heldout']:.4f} (measured 0.9775)  {'PASS' if gD else 'FAIL'}")
# round-trip through pickle, since that is how the trainer will load it
import pickle,tempfile,os
p=os.path.join(tempfile.mkdtemp(),"probe.pkl"); srp.save(probe,p)
rt=srp.load(p)
s1=probe.predict_proba(X[te][:64],text=[texts[i] for i in np.where(te)[0][:64]])[:,1]
s2=rt.predict_proba(X[te][:64],text=[texts[i] for i in np.where(te)[0][:64]])[:,1]
gE=np.allclose(s1,s2,atol=0,rtol=0)
print(f"GATE E  save/load round-trip identical : max|diff|={np.abs(s1-s2).max():.3e}  {'PASS' if gE else 'FAIL'}")
print(f"\n{'ALL GATES PASS' if all([gA,gB,gC,gD,gE]) else 'GATE FAILURE - DO NOT LAUNCH'}")
