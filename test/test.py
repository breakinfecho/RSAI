from openai import OpenAI
from evl_e import cal
from evl_c import calc
import json
import pickle


def getOut(can, rate):
    try:
        completion = client.chat.completions.create(
            model="/data/njf/RSAI",
            # model="/data/njf/Hermes3",
            # model="/data/share/Llama-3-8B-instruct",
            # model="/data/njf/Astrops",
            # model="/data/share/models/Qwen2.5-7B-Instruct",
            #  model="/data/share/models/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "user", "content": "You are a professional text interpreter."},
                {"role": "user",
                 "content": "You need to predict the survival prediction of the restaurant (0-dead, 1-alive) (If you think "
                            "the reviews are too one-sided or aggressive, consider the restaurant average rating as mentioned "
                            "above) and explain this review if you are the first-hand perspective "
                            "of the reviewer(do not mention rating and survival in explanation)."
                            " Your output should strictly be-'prediction: .explanation: .'within 50 words"
                 },
                {"role": "assistant",
                 "content": "Ok, I will do it considering reviews and restaurant's rating.！And I do not generate any "
                            "own thinking process and useless information "},
                {"role": "user",
                 "content": "Here is a review about a restaurant:" + can +
                            "and its average rate (all reviews average rate): " + rate + "Notice!Do not "
                                                                                         "generate"
                                                                                         "any useless information"},
                # {"role": "user",
                #  "content": "Give this restaurant some personalized recommendations based on this review"}
            ],
            max_tokens=256,
        )
        import re
        print(completion)
        # 使用正则表达式提取第一个 prediction 和对应的 explanation
        match = re.search(r'prediction:\s*(\d+)\s*explanation:\s*([^\<]*)', completion.choices[0].message.content,
                          re.IGNORECASE)

        # match = re.search(r'prediction:\s*(\d+)\.\s*explanation:\s*(.+)', completion.choices[0].message.content,  re.IGNORECASE)
        if match:
            prediction = match.group(1)
            explanation = match.group(2).strip()
            if prediction == " " or explanation == "" or explanation == ".":
                return -1
            return {
                "prediction": prediction,
                "explanation": explanation
            }
    except Exception as e:
        print(f"Error in getOutput: {e}")

    return -1


client = OpenAI(
    base_url="http://127.0.0.1:50071/v1",
    api_key="123456",
)
print("服务连接成功")
count = 0
correct = 0
bleu = 0.0
rouge1 = 0.0
rouge2 = 0.0
rougel = 0.0
rougeSu4 = 0.0
try:
    with open('/home/njf/Rec/data/ON/dataset_evl.pickle', 'rb') as f_check:
        loaded_data = pickle.load(f_check)

    for i in range(1, 1000):
        can = loaded_data[0][i]['input_field_data']['review']
        rate = loaded_data[0][i]['input_field_data']['rating']
        pre = loaded_data[0][i]['input_field_data']['survival']
        res = getOut(can, rate)
        print(i)
        print(can)
        print(res)
        if res != -1:
            count += 1
            if pre == res['prediction']:
                correct += 1
            ans = cal(can, res['explanation'])
            bleu += ans['BLEU']
            rouge1 += ans['ROUGE-1']
            rouge2 += ans['ROUGE-2']
            rougel += ans['ROUGE-L']
            rougeSu4 += ans['ROUGE-SU4']
            print(ans)
            print('-----------------------------')


except FileNotFoundError as e:
    print(count)
    print(f"Accuracy: {correct / count}")
    print(f"Average BLEU: {bleu / count}")
    print(f"Average ROUGE-1: {rouge1 / count}")
    print(f"Average ROUGE-2: {rouge2 / count}")
    print(f"Average ROUGE-L: {rougel / count}")
    print(f"Average ROUGE-SU4: {rougeSu4 / count}")

# 输出最终结果
print(f"Total count: {count}")
if count > 0:
    print(f"Accuracy: {correct / count}")
    print(f"Average BLEU: {bleu / count}")
    print(f"Average ROUGE-1: {rouge1 / count}")
    print(f"Average ROUGE-2: {rouge2 / count}")
    print(f"Average ROUGE-L: {rougel / count}")
    print(f"Average ROUGE-SU4: {rougeSu4 / count}")
else:
    print("No valid predictions were made.")
