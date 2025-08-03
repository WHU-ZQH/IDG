# IDG

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

generated data in pseudo_data/
