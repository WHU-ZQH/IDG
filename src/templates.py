
EX_template = '''We will now perform an Aspect-Based Sentiment Analysis task. In this task, you are required to:
- Identify the aspects mentioned in the text 
- Determine the sentiment polarity toward each aspect (positive, neutral, or negative)
- Output format: [aspect, sentiment_polarity]
{example}
Now, complete the aspect extraction task for the text below:
Input: "{input}"
Output: '''

ET_template = '''We will now perform an Aspect-Based Sentiment Analysis task, in this task, Expansion of homonyms or synonyms for a given aspectual word
Generation of 5-10 cognate or synonymous aspect words for an aspect word
example:
input:{example_in}
output:{example_out}
Now, complete the aspect extend task for the text below:
input: {input} 
output:'''

Eval_filter = '''You need to perform a task of sentiment judgment and domain judgment, the task requirements are shown below:
- Determine whether the potential sentiment hidden in the sentence by aspect is positive, negative, or neutral based on the context given in the sentence.
- Avoid confusing the neutral sentiment of the aspect with a positive or negative sentiment.
- Is this sentence related to {domain} ? If so, output “Y”; otherwise, output “N”.
- Just give the answer without any unnecessary explanations.

Now, please complete the task for the following input:
- input format: sentence #aspect
- output format: sentiment; Y(N)
Input: {input}
Output:'''

# Eval_score = '''{example}
# You are a psycholinguist who analyses sentiment and scores the above sentences in the
# following three areas:
# 1. Possessing complex syntactic structures, such as inverted sentences, imperative sen-
# tences, sentences with inflections, and sentences beginning with multiple combinations of
# adverbs, nouns, and subjects, the more complex the higher the score.
# 2. With a rich vocabulary, the richer the score, the higher the score.
# 3. User comments that match real-life scenarios, the more they match, the higher the score.

# Please provide an accurate score between 1 and 10 for each aspect, and finally, output an overall average score. The output format should be as follows:
# Sample X.X: [Syntax: score; Vocabulary: score; Real-life relevance: score; Overall score: **score**]
# All scores should be output in two decimal places.'''

Eval_score = '''{example}
You are a psycholinguist who analyses sentiment and scores the above sentences in the
following three areas:
1. Possessing complex syntactic structures, such as inverted sentences, imperative sen-
tences, sentences with inflections, and sentences beginning with multiple combinations of
adverbs, nouns, and subjects, the more complex the higher the score.
2. With a rich vocabulary, the richer the score, the higher the score.
3. User comments that match real-life scenarios, the more they match, the higher the score.

Please provide an accurate score between 1 and 10 for each aspect, and finally, output an overall average score. The output format should be as follows:
Sample X.X: [Syntax: score; Vocabulary: score; Real-life relevance: score; Overall score: **score**]
All scores should be output in two decimal places.
You don't need to give any unnecessary explanations. Just output it in the given format.'''

# ITAT_template = '''
# As a product user in the {domain} field, we would like you to complete a creative sentence generation task.
# Please be as imaginative as possible to diversify your comments.
# Please follow these requirements:
# - Teaching analysis – analyzing the given aspect and sentiment:
# - Specify the sentiment of the aspect in the generated sample.
# - Generate a sentence containing exactly {aspect_num} aspects, clarify the meaning of the aspect, and generate sentences corresponding to the polarity of the sentiment.
# - The generated sentence must be in length within {length} words.
# - as if the speaker is describing their personal experience or opinion.
# - Use more laid-back, everyday language, or shift the tone to make it feel more relaxed.
# - The generated sentence must contain exactly {aspect_num} aspects and each aspect must be clearly mentioned.
# - Generated sentences can contain only one period at a time and must strictly follow the example format with the specified number of aspects.
# Input: {example_input}
# Output: {example_output}
# Please imitate the sentence structure from the example.
# Now, complete this task in a natural human-like manner and generate only one sentence:
# Input: {input}
# Output:
# '''

ITAT_zeroshot_template = '''
We would like you to complete a sentence generation task, , and we will tell you how to
generate appropriate sentences. Please follow these requirements:
-Teaching analysis - analyzing the given aspect and sentiment:
- Specify the sentiment of the aspect in the generated sample.
- Domain of sample generation: {domain}
- Generate a sentence containing a given aspect, clarify the meaning of the aspect, and generate sentences corresponding to the polarity of the sentiment.
- The generated sentence must be in length within {length} words.
- Generated sentences can contain only one period at a time and the sentence should not consist of an unspecified aspect.
Now, complete this task in a natural human-like manner and generate only one sentence:
Input: {input}
Output:
'''

# gpt-3.5-turbo ITAT_template

ITAT_template = '''We would like you to complete a sentence generation task, and we will tell you how to generate appropriate sentences. Please follow these requirements:
- Teaching analysis - analyzing the given aspect and sentiment:
- Specify the sentiment of the aspect in the generated sample.
- Domain of sample generation: {domain}
- Generate a sentence containing a given aspect, clarify the meaning of the aspect, and generate sentences corresponding to the polarity of the sentiment.
- The generated sentence must be in length within {length} words.
- Generated sentences can contain only one period at a time and the sentence should not consist of an unspecified aspect.
Examples:
input: {example_input}
output: {example_output}
Now, complete this task in a natural human-like manner and generate only one sentence:
Input: {input}
Output:'''
