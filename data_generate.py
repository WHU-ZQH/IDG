import os
import re
import random
import argparse
import jsonlines

from src.api import *
from src.templates import *
from src.data_utils import process_orig_data, format_string
from datetime import datetime


def aspect_sentiment_comb_mix(args):
    aspect_sentiment_set = []   # ['asp,sen;asp,sen;...', ...]
    aspect_sentiment_list = []  # ['asp,sen;', ...]
    selected_sentiment = args.sentiment_polarity.copy()
    assert len(selected_sentiment) == len(args.aspect_set_path)
    for i in range(len(selected_sentiment)):
        file_path = args.aspect_set_path[i]
        sentiment = selected_sentiment[i]
        with open(file_path,'r',encoding='utf-8') as asp:
            for aspect in asp:
                aspect_sentiment = aspect.split('\n')[0] + ',' + sentiment + ';'
                aspect_sentiment_list.append(aspect_sentiment)

    random.shuffle(aspect_sentiment_list)
    aspect_sentiment_str = ''
    num = 0
    for aspect_sentiment in aspect_sentiment_list:
        aspect_sentiment_str += aspect_sentiment
        num += 1
        if num == args.aspect_num:
            aspect_sentiment_set.append(aspect_sentiment_str)
            aspect_sentiment_str = ''
            num = 0
    
    num_samples = len(aspect_sentiment_set)
    num_batches = num_samples // args.batch_size
    dataset = []    # [['asp,sen;asp,sen;...', ...], ...]
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx + 1) * args.batch_size
        batch_data = aspect_sentiment_set[start_idx:end_idx]
        dataset.append(batch_data)
    remainder_data = aspect_sentiment_set[num_batches * args.batch_size:]
    if remainder_data:
        dataset.append(remainder_data)
    
    return dataset


def format_generate_instructions(aspect_sentiment_set, examples, args):
    template = ITAT_template
    instructions = []
    
    for inp in aspect_sentiment_set:
        if len(examples) == 0:
            example = random.choice(orig_examples)
        else:
            example = random.choice(examples)
        
        sentence, tags = example.split('#')  # 分离出句子和标签部分
        ex_input = format_string(tags)
        ex_output = sentence.strip().strip('[]')
        instructions.append(template.format(
            input=inp,
            domain=args.domain,
            length=args.length,
            example_input=ex_input,
            example_output=ex_output)
        )
    
    return instructions


def generate_samples(instructions, args):
    all_res = []
    for query in instructions:
        res = invoke_gpt_generate(query, args.model_name)
        all_res.append(res)
    return all_res


def check_aspect_num(result, aspect_sentiment_set, args):
    """
    检查生成的句子中是否包含指定数量的aspect，并将其替换为$T$标记。
    """
    pseudo_samples = []  # ['xxx $T$ xxx#asp,sen', ...]
    aspects = []
    for aspect_sentiment in aspect_sentiment_set[:-1].split(';'):
        aspect = aspect_sentiment.split(',')[0]
        aspects.append(aspect)
        if '\n' in result:
            continue
        if aspect.lower() in result.lower():
            sample_line = re.sub(rf"\b{re.escape(aspect)}\b", "$T$", result, flags=re.IGNORECASE) + '#' + aspect_sentiment
            pseudo_samples.append(sample_line)
    if len(pseudo_samples) < args.aspect_num:
        with open(args.aspect_num_error_log_path, 'a', encoding='utf-8') as f:
            f.write(result + f"#{','.join(aspects)}\n")
        return False
    else:
        return pseudo_samples


def check_sentiments(pseudo_samples, args):
    """
    检查生成的句子领域是否相关、情感是否正确，并返回符合条件的样本。
    """
    samples = []
    domain_num = 0
    for line in pseudo_samples:
        # line: 'xxx $T$ xxx#asp,sen'
        eval_filter_template = Eval_filter
        true_sentiment = line.split('#')[1].split(',')[1].split(']')[0].strip()
        true_aspect = line.split('#')[1].split(',')[0]
        sentence = line.split('#')[0].replace("$T$", true_aspect)
        input = f'sentence: {sentence} #aspect: {true_aspect}'
        instruction = eval_filter_template.format(input=input, domain=args.domain)

        try:
            result = invoke_gpt_sentiment(instruction, args.model_name)
            pred_sentiment = re.search(r'\b(positive|negative|neutral)\b', result.lower()).group(1)
            domain_accuracy = re.search(r'\b(Y|N)\b', result).group(1)
            if domain_accuracy == 'Y':
                domain_num += 1
            if pred_sentiment == true_sentiment:
                samples.append(line)
        except Exception as e:
            print(f'eval error: {e}')
    
    if domain_num == 0:
        # 领域不相关
        with open(args.domain_error_log_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(pseudo_samples) + "\n")
    elif len(samples) < args.aspect_num:
        # 任意aspect情感不正确
        with open(args.sentiment_error_log_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(pseudo_samples) + "\n")
    else:
        return samples


def pseudo_sample_generate(instructions, aspect_sentiment_set, args):
    results = generate_samples(instructions, args)
    # print(results)
    pseudo_samples = []
    for i in range(len(results)):
        try:
            pseudo_samples_ = check_aspect_num(results[i], aspect_sentiment_set[i], args)
            if pseudo_samples_:
                eval_signal = check_sentiments(pseudo_samples_, args)
                if eval_signal:
                    for line in eval_signal:
                        pseudo_samples.append(line)
        except Exception as e:
            print(e)
    return pseudo_samples


def select(pseudo_sample, step, args):
    qualified_samples = open(args.qualified_sample_path, 'a', encoding='utf-8')
    discarded_samples = open(args.discarded_sample_path, 'a', encoding='utf-8')
    eval_score_high = Eval_score
    
    # 将同一句话的不同aspect和sentiment组合成一句话
    example_str = ''
    select_p = []   # ['sentence#asp1,sen1;asp2,sen2', ...]
    mutl_fl = 0
    for i, line in enumerate(pseudo_sample):
        # line: 'xxx $T$ xxx#asp,sen'
        if mutl_fl == 0:
            true_sentiment = line.split('#')[1].split(',')[1].replace(']', '').strip()
            true_aspect = line.split('#')[1].split(',')[0]
            sentence = line.split('#')[0].replace("$T$", true_aspect).replace('[', '')
            line_p = sentence + '#' + true_aspect + ',' + true_sentiment
            mutl_fl += 1
            if args.aspect_num == 1:
                select_p.append(line_p)
                example_str += f'sample {(i + 1) / args.aspect_num}: "{line_p}"\n'
                mutl_fl = 0
        elif mutl_fl < args.aspect_num - 1 :
            true_sentiment = line.split('#')[1].split(',')[1].replace(']', '').strip()
            true_aspect = line.split('#')[1].split(',')[0]
            line_p = line_p + ';' + true_aspect + ',' + true_sentiment
            mutl_fl += 1
        else:
            mutl_fl = 0
            true_sentiment = line.split('#')[1].split(',')[1].replace(']', '').strip()
            true_aspect = line.split('#')[1].split(',')[0]
            line_p = line_p + ';' + true_aspect + ',' + true_sentiment
            select_p.append(line_p)
            example_str += f'sample {(i + 1)/args.aspect_num}: "{line_p}"\n'
    # print(f'qualified num: {len(select_p)}')
    
    if len(select_p) == 0:
        print(f'No qualified samples found in step {step + 1}.')
        return [], []
    
    # 评分筛选
    instruction = eval_score_high.format(example=example_str)
    sample_score_dict = {}
    try:
        score_example = invoke_gpt_sentiment(instruction, args.model_name)
        high_quality_samples = []
        score_example = score_example.split('\n')
        score_example = [line.strip() for line in score_example if line.strip()]
        assert len(score_example) == len(select_p), f'The number of evaluations output is inconsistent with the number of inputs.\nselect_p: {select_p}\nscore_example: {score_example}'
        
        for i, example_line in enumerate(score_example):
            comprehensive_score_match = re.search(r'Overall score:\s*\*{0,2}(\d+\.\d{2})\*{0,2}', example_line)
            if comprehensive_score_match:
                comprehensive_score = float(comprehensive_score_match.group(1))
            else:
                print(f'not match {i}: {example_line}')
                continue
            
            if comprehensive_score >= args.threshold:
                sample_score_dict[i] = comprehensive_score
                for j in range(args.aspect_num):
                    high_quality_samples.append(pseudo_sample[i * args.aspect_num + j])
            else:
                for j in range(args.aspect_num):
                    aspect_sentiment = pseudo_sample[i * args.aspect_num + j].split('#')[1].replace(']','')
                    discarded_samples.write(f"{aspect_sentiment}\n")
    except Exception as e:
        print(f"select error: {e}")
    
    sorted_samples = sorted(sample_score_dict.items(), key=lambda x: x[1], reverse=True)
    high_samples = [f'[{select_p[i[0]]}]' for i in sorted_samples]
    
    qualified_res = f"step: {step + 1}\n"
    for i in range(len(select_p)):
        qualified_res += f'sample {i + 1}: "{select_p[i]}"\nscore: {score_example[i]}\n'
    qualified_samples.write(qualified_res)
    
    print(f'save num: {len(high_quality_samples)}')
    
    return high_samples, high_quality_samples


def save_mix_pseudo_samples(result_pseudo_samples, args):
    with jsonlines.open(args.save_path, mode='a') as writer:
        for line in result_pseudo_samples:
            pseudo_sample = {'sentence': [], 'aspect': [], 'sentiment': []}
            sentiment = line.split('#')[1].split(',')[1].replace(']','').strip()
            aspect = line.split('#')[1].split(',')[0]
            sentence = line.split('#')[0].replace('[','')
            pseudo_sample['sentence'] = sentence
            pseudo_sample['aspect'] = aspect
            pseudo_sample['sentiment'] = sentiment
            writer.write(pseudo_sample)


def process_data_in_batches(dataset, args):
    gen_num = 0
    examples = []    # 初始示例
    for idx, aspect_sentiment_set in enumerate(dataset):
        print('Processing batch:', idx + 1)
        # aspect_sentiment_set: ['asp1,sen1;asp2,sen2;...', ...]
        # instruntions: ['...', ...]
        instruntions = format_generate_instructions(aspect_sentiment_set, examples, args)
        # result_pseudo_sample: ['xxx $T$ xxx#asp,sen', ...]
        result_pseudo_samples = pseudo_sample_generate(instruntions, aspect_sentiment_set, args)
        # example: 
        # high_score_pseudo_sample: 
        examples, high_score_pseudo_samples = select(result_pseudo_samples, idx, args)
        save_mix_pseudo_samples(high_score_pseudo_samples, args)
        
        gen_num += len(high_score_pseudo_samples)
        if args.max_gen_num > 0 and gen_num >= args.max_gen_num:
            print(f"Reached maximum generation limit: {args.max_gen_num}. Stopping further generation.")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt-4o-mini', type=str)
    parser.add_argument('--aspect_num', default=1, type=int)
    parser.add_argument('--max_gen_num', type=int, default=-1, help='Maximum amount of data generated')
    parser.add_argument('--length', type=int, default=20, help='Length of output')
    parser.add_argument('--batch_size',type=int,default=12)
    parser.add_argument('--threshold', type=float, default=6.0, help='Threshold of discriminator')
    parser.add_argument('--examples', type=str, default='laptop14')
    parser.add_argument('--output_dir', type=str, default='laptop')
    parser.add_argument('--domain', type=str, default='laptop', help='Domain (e.g., restaurant)')
    parser.add_argument('--sentiment_polarity', nargs='+', default=['negative', 'neutral', 'positive'],help='List of sentiment polarities')
    parser.add_argument('--aspect_set_path', nargs='+', default=[
        './aspect_set/laptop14/laptop_process_neg.txt',
        './aspect_set/laptop14/laptop_process_neu.txt',
        './aspect_set/laptop14/laptop_process_pos.txt'],
                        help='List of paths to aspect sets')
    args = parser.parse_args()
    
    # 初始示例
    if args.examples == 'laptop14':
        orig_path = 'original_data/laptop14/unlabeled_corpus.txt'
    elif args.examples == 'rest14':
        orig_path = 'original_data/rest14/unlabeled_corpus.txt'
    elif args.examples == 'rest15':
        orig_path = 'original_data/rest15/unlabeled_corpus.txt'
    elif args.examples == 'rest16':
        orig_path = 'original_data/rest16/unlabeled_corpus.txt'
    else:
        raise ValueError("Invalid examples argument. Choose from 'laptop14', 'rest14', 'rest15', or 'rest16'.")
    orig_examples = process_orig_data(orig_path)[args.aspect_num]
    random.shuffle(orig_examples)
    orig_examples = orig_examples[:20]
    
    current_time = datetime.now().strftime("%m%d_%H%M")
    out_path = f'./out/{args.output_dir}/aspect_{args.aspect_num}_model_{args.model_name}/length_{args.length}_threshold_{args.threshold}_{current_time}'
    os.makedirs(out_path, exist_ok=True)
    args.aspect_num_error_log_path = os.path.join(out_path, 'aspect_num_error_log.txt')
    args.domain_error_log_path     = os.path.join(out_path, 'domain_error_log.txt')
    args.sentiment_error_log_path  = os.path.join(out_path, 'sentiment_error_log.txt')
    args.qualified_sample_path     = os.path.join(out_path, 'qualified_samples.txt')
    args.discarded_sample_path     = os.path.join(out_path, 'discarded_samples.txt')
    args.save_path                 = os.path.join(out_path, 'data.jsonl')  # 最终生成结果路保存路径
    
    # compose aspect-sentiment pairs
    aspect_sentiment_set_mix = aspect_sentiment_comb_mix(args)
    # generate pseudo samples
    process_data_in_batches(aspect_sentiment_set_mix, args)
    
    print(f"Data generation completed. Results saved to {args.save_path}")
