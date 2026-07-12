from transformers import Mask2FormerForUniversalSegmentation
m = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-mapillary-vistas-semantic")
id2label = m.config.id2label
print("num classi:", len(id2label))
for i in sorted(id2label):
    if any(k in id2label[i].lower() for k in ["road","lane","crosswalk","curb","sidewalk","bike","service","marking","parking"]):
        print(i, "::", id2label[i])