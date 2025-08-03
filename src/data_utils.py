import ast
import logging

from collections import defaultdict


def process_orig_data(input_file):
    # 使用字典按词组列表长度分类存储结果
    results = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 分割原始句子和词组列表
            if "####" not in line:
                continue
                
            sentence, phrase_data = line.split("####", 1)
            sentence = sentence.strip()
            
            try:
                # 解析词组列表
                phrase_list = ast.literal_eval(phrase_data.strip())
            except (SyntaxError, ValueError):
                continue
            
            # 获取单词列表
            words = sentence.split()
            processed_phrases = []
            
            # 处理每个词组
            for indices, label in phrase_list:
                # 提取词组单词
                phrase_words = [words[i] for i in indices if i < len(words)]
                phrase_str = " ".join(phrase_words)
                
                # 转换标签
                label_map = {
                    'POS': 'positive',
                    'NEG': 'negative',
                    'NEU': 'neutral'
                }
                processed_label = label_map.get(label, label.lower())
                
                # 拼接词组和标签
                processed_phrases.append(f"{phrase_str},{processed_label}")
            
            # 构建最终结果字符串
            result_str = f"{sentence}#{';'.join(processed_phrases)}"
            
            # 按词组列表长度分类存储
            results[len(phrase_list)].append(result_str)
    
    return results


TAG_TO_WORD = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
NONE_TOKEN = "[none]"
ASPECT_TOKEN = "<aspect>"
OPINION_TOKEN = "<opinion>"
TAG_TO_SPECIAL = {"POS": ("<pos>", "</pos>"), "NEG": ("<neg>", "</neg>"), "NEU": ("<neu>", "</neu>")}
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
logger = logging.getLogger(__name__)

def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    logger.info(f"{data_path.split('/')[-1]}\tTotal examples = {len(sents)} ")
    return sents, labels


def get_inputs(args, data_type_file="train"):
    """
        train_inputs: ["hi", "I love apples."],
    """
    data_path = f"{args.data_dir}/{data_type_file}.txt"
    inputs, _ = read_line_examples_from_file(data_path)
    inputs = [" ".join(i) for i in inputs]
    return inputs

def prepare_EX_extraction(data_path):
    sents, labels = read_line_examples_from_file(data_path)
    inputs = [" ".join(s) for s in sents]

    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                # single aspect
                if len(tri[0]) == 1:
                    try:
                        a = sents[i][tri[0][0]]
                    except:
                        continue
                else:
                    start_idx, end_idx = tri[0][0], tri[0][-1]
                    a = ' '.join(sents[i][start_idx:end_idx+1])
                try:
                    c = TAG_TO_WORD[tri[1]]
                except:
                    c = TAG_TO_WORD[tri[2]]
                all_tri.append((a, c))
            label_strs = ['['+', '.join(l)+']' for l in all_tri]
            targets.append('; '.join(label_strs))
    return inputs, targets

def format_string(input_string):
    # 移除末尾多余的字符，比如']'
    input_string = input_string.strip(']')

    # 按照';'分割字符串，并去除每个元素的前后空格
    parts = [part.strip() for part in input_string.split(';') if part.strip()]

    # 为每个部分添加引号和逗号
    formatted_parts = [f"'{part};'" for part in parts]

    # 将所有部分组合成最终的格式
    formatted_string = "[" + ",".join(formatted_parts) + "]"

    return formatted_string


if __name__ == "__main__":
    input_file = "original_data/laptop14/unlabeled_corpus.txt"

    processed_data = process_orig_data(input_file)
    
    for length, items in processed_data.items():
        print(f"长度 {length} 的数据有 {len(items)} 条")
        print(items[:3])
