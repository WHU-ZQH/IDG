import time
import warnings

from openai import OpenAI
from tenacity import (retry, stop_after_attempt, wait_random_exponential)

warnings.filterwarnings("ignore")

# chatanywhere
client = OpenAI(
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://api.chatanywhere.tech/v1",
)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def invoke_gpt_generate(prompt, model_engine, temperature=1.0):
    retry_count = 5
    retry_interval = 1

    for _ in range(retry_count):
        try:
            completion = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "You are a critic who can generate comments on specified aspect and sentiment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except TimeoutError:
            print("Timeout: ", prompt)
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
        except Exception as e:
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
            print(e)
            print(prompt)
    
    return ""


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def invoke_gpt_sentiment(prompt, model_engine, temperature=0, top_p=0.7):
    retry_count = 5
    retry_interval = 1
    
    for _ in range(retry_count):
        try:
            completion = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "You are an AI assistant specializing in linguistics and sentiment analysis. "},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                extra_body={"enable_thinking": False},
            )
            return completion.choices[0].message.content
        except TimeoutError:
            print("Timeout: ", prompt)
            retry_interval *= 2
            time.sleep(retry_interval)
        except Exception as e:
            retry_interval *= 2
            time.sleep(retry_interval)
            print(e)
            print(prompt)
            
    return ""


if __name__ == "__main__":
    prompt = "Generate a comment on the aspect of 'plot' with a positive sentiment for a movie."
    res = invoke_gpt_generate(prompt, 'gpt-4o-mini')
    print(res)
