from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch, torch.nn.functional as F, numpy as np, cv2
from PIL import Image

m = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
proc = SegformerImageProcessor.from_pretrained(m)   # resize/normalize di default, NON toccati
model = SegformerForSemanticSegmentation.from_pretrained(m).eval()

cap = cv2.VideoCapture("/Users/robertodioguardi/Desktop/pie_video/set01/video_0001.mp4")
ok, f = cap.read(); cap.release()
img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

inp = proc(images=img, return_tensors="pt")
with torch.no_grad():
    logits = model(**inp).logits
up = F.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
pred = up.argmax(1)[0].numpy()

# cosa c'e' davvero nella maschera:
ids, counts = np.unique(pred, return_counts=True)
tot = pred.size
for i, c in zip(ids, counts):
    print(f"classe {i}: {100*c/tot:.1f}%")