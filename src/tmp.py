from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, pandas as pd
from config import *
from af_front_pipeline_stages import _log_ppl
device = torch.device('cuda')
tok = AutoTokenizer.from_pretrained(LM_MODEL)
lm  = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
df  = pd.read_csv(SPLITS_DIR / 'emea_dev.csv').dropna().head(200)
scores = [-_log_ppl(r, lm, tok) for r in df.ref_fr if _log_ppl(r, lm, tok)]
print(min(scores), max(scores))