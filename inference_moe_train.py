#!/usr/bin/env python3
"""inference_moe.py แต่ชี้ไปที่ train_set.json"""
from inference_moe import *

TEST_PATH  = "data/ชุดข้อมูล/train_set.json"
OUTPUT_CSV = "submission_train.csv"

if __name__ == "__main__":
    import inference_moe as m
    m.TEST_PATH  = TEST_PATH
    m.OUTPUT_CSV = OUTPUT_CSV
    m.main()
