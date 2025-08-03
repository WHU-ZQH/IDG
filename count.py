import os


def count_categories(file_path):
    category_counts = {'-1': 0, '0': 0, '1': 0}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        assert len(lines) % 3 == 0
        
        for i in range(2, len(lines), 3):
            category = lines[i].strip()
            assert category in ['-1', '0', '1']
            category_counts[category] += 1
    
    print(f'in {file_path}:')
    print('| pos   | neu   | neg   | sum   |')
    print(f'| {category_counts['1']}  | {category_counts["0"]}   | {category_counts['-1']}  | {category_counts['1'] + category_counts['0'] + category_counts['-1']}  |')


if __name__ == "__main__":
    # root = 'generated_data'
    # for data in ['IterD', 'gpt-3.5-turbo', 'gpt-4o-mini', 'deepseek-v3', 'qwen-plus']:
    #     for file in ['laptop14_generated.raw', 'rest14_generated.raw', 'rest15_generated.raw', 'rest16_generated.raw']:
    #         count_categories(os.path.join(root, data, file))
    
    root = 'pseudo_data'
    for file in ['laptop14_generated.raw', 'rest14_generated.raw', 'rest15_generated.raw', 'rest16_generated.raw']:
        count_categories(os.path.join(root, file))
