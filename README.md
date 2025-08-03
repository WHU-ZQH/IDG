# Iterative Data Generation with Large Language Models for Aspect-based Sentiment Analysis
----

## Requirements
----
* Python 3.10
* jsonlines
* tqdm
* nltk
* openai 0.28.2

## Dataset 
We conduct the main experiments on 4 widely-used ABSA benchmarks: Laptop14, Restaurant14, Restaurant15 and Restaurant16.

- Original data: "./original_data/"
- Generated data: "./pseudo_data/"

## How to run
- 
IDG data generation: 

``` sh
python data_generate.py \
    --model_name "gpt-3.5-turbo" \
    --aspect_num 1 \
    --length 15 \
    --threshold 6.0 \
    --examples "laptop14" \
    --output_dir "laptop" \
    --domain "laptop" \
    --aspect_set_path "./aspect_set/laptop14/laptop_process_neg.txt" \
                      "./aspect_set/laptop14/laptop_process_neu.txt" \
                      "./aspect_set/laptop14/laptop_process_pos.txt"
```

## Citation

If you find this useful in your research, please consider citing:

```
@article{zhong2024iterative,
  title={Iterative data generation with large language models for aspect-based sentiment analysis},
  author={Zhong, Qihuang and Li, Haiyun and Gong, Jinzhao and Liu, Juhua and Du, Bo},
  journal={arXiv preprint arXiv:2407.00341},
  year={2024}
}
```
