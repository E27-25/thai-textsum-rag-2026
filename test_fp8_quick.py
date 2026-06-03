#!/usr/bin/env python3
"""Quick sanity-check: run first 50 queries with FP8 single-GPU (submission mode)."""
import json, tempfile, os
import inference_moe as m

with open("data/ชุดข้อมูล/train_set.json", encoding="utf-8") as f:
    data = json.load(f)
data["queries"] = data["queries"][:50]

tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
json.dump(data, tf, ensure_ascii=False)
tf.close()

m.TEST_PATH  = tf.name
m.OUTPUT_CSV = "submission_fp8_test.csv"

m.main()
os.unlink(tf.name)
print("FP8 quick-test done.")
