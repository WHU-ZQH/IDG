import json

def convert_sentiment(sentiment):
    """将情感标签转换为数字表示"""
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return '1'
    elif sentiment == 'neutral':
        return '0'
    elif sentiment == 'negative':
        return '-1'
    else:
        raise ValueError(f"未知情感标签: {sentiment}")

def process_jsonl_files(input_files, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        processed_count = 0
        skipped_count = 0
        
        for file_path in input_files:
            print(f"处理文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as in_f:
                lines = {line for line in in_f.readlines()}
                for line in lines:
                    try:
                        data = json.loads(line)
                        
                        # 提取字段
                        sentence = data.get('sentence', '').strip().replace("\n", " ")
                        aspect = data.get('aspect', '').strip()
                        sentiment = convert_sentiment(data.get('sentiment', ''))
                        
                        # 跳过无效数据
                        if not all([sentence, aspect]) or sentiment == 'unknown':
                            skipped_count += 1
                            continue
                        
                        # 写入三行格式
                        out_f.write(sentence + '\n')
                        out_f.write(aspect + '\n')
                        out_f.write(sentiment + '\n')
                        
                        processed_count += 1
                    
                    except json.JSONDecodeError:
                        skipped_count += 1
                        continue
        
        print(f"处理完成! 成功处理 {processed_count} 条记录")
        if skipped_count > 0:
            print(f"跳过 {skipped_count} 条无效记录")


if __name__ == '__main__':
    process_jsonl_files([
        'out/laptop/aspect_1_model_gpt3.5/length_15_threshold_6.0_0708_1602/data.jsonl',
        'out/laptop/aspect_2_model_gpt3.5/length_20_threshold_6.0_0708_1902/data.jsonl',
        'out/laptop/aspect_3_model_gpt3.5/length_25_threshold_6.0_0708_1902/data.jsonl',
    ], './pseudo_data/laptop14_generated.raw')
    process_jsonl_files([
        'out/rest14/aspect_1_model_gpt3.5/length_15_threshold_6.0_0708_1902/data.jsonl',
        'out/rest14/aspect_2_model_gpt3.5/length_20_threshold_6.0_0708_1902/data.jsonl',
        'out/rest14/aspect_3_model_gpt3.5/length_25_threshold_6.0_0708_1902/data.jsonl',
    ], './pseudo_data/rest14_generated.raw')
    process_jsonl_files([
        'out/rest15/aspect_1_model_gpt3.5/length_15_threshold_6.0_0709_0951/data.jsonl',
        'out/rest15/aspect_2_model_gpt3.5/length_20_threshold_6.0_0709_1600/data.jsonl',
        'out/rest15/aspect_3_model_gpt3.5/length_25_threshold_6.0_0709_1600/data.jsonl',
    ], './pseudo_data/rest15_generated.raw')
    process_jsonl_files([
        'out/rest16/aspect_1_model_gpt3.5/length_15_threshold_6.0_0708_2228/data.jsonl',
        'out/rest16/aspect_2_model_gpt3.5/length_20_threshold_6.0_0708_2228/data.jsonl',
        'out/rest16/aspect_3_model_gpt3.5/length_25_threshold_6.0_0709_0951/data.jsonl',
    ], './pseudo_data/rest16_generated.raw')
