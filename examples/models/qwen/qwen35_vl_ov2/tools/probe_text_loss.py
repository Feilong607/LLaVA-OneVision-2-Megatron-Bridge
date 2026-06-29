#!/usr/bin/env python3
"""Pure-text loss probe of the frozen converted Qwen3.5-35B-A3B *text* LLM.
Purpose: localize the stage-1 adapter-only +1.1-nat plateau gap vs 30B.
A healthy 35B base => low CE on coherent prose (~2). If ~3.7+/garbage => extracted weights broken.
Runs the HF model (transformers 5.8.1 supports qwen3_5_moe_text) sharded across all GPUs."""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MP = os.environ.get("TEXT_DIR", "/ov2/pretrain_models/Qwen3.5-35B-A3B-text")
print("loading", MP, flush=True)
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
dev0 = next(model.parameters()).device
print("loaded ok; first-param device =", dev0, flush=True)

coherent = [
  "The mitochondria is the powerhouse of the cell. It generates most of the cell supply of adenosine triphosphate, used as a source of chemical energy.",
  "In 1969, the Apollo 11 mission successfully landed the first humans on the Moon. Neil Armstrong became the first person to step onto the lunar surface.",
  "Machine learning is a subfield of artificial intelligence that focuses on building systems that learn from data. Deep neural networks have driven much of the recent progress.",
  "Water is composed of two hydrogen atoms and one oxygen atom. At standard temperature and pressure, it exists as a clear, colorless liquid.",
  "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect against nomadic invasions.",
]
captions = [
  "a photo of a dog playing in the park",
  "a red car parked on a city street at night",
  "two people sitting on a wooden bench by the sea",
  "a close up of a plate of pasta with tomato sauce",
  "a black and white cat sleeping on a sofa",
  "an aerial view of a green forest and a winding river",
  "a young child holding a colorful balloon",
  "a cup of coffee on a wooden table next to a book",
]

def group_loss(name, texts):
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.to(dev0)
            if ids.shape[1] < 2: continue
            out = model(input_ids=ids, labels=ids)
            n = ids.shape[1]-1
            tot += out.loss.item()*n; ntok += n
            print("   ppl-tok=%.3f n=%d :: %s" % (out.loss.item(), n, t[:50]), flush=True)
    print("[%s] MEAN per-token CE = %.4f  (over %d tokens)" % (name, tot/ntok, ntok), flush=True)

group_loss("coherent-english", coherent)
group_loss("short-captions", captions)
print("=== PROBE DONE ===", flush=True)
