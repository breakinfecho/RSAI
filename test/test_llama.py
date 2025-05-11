from openai import OpenAI
from evl_e import cal
from evl_c import calc
import json
import pickle


def getOut(can, rate):
    try:
        completion = client.chat.completions.create(
            model="/data/share/Llama-3-8B-instruct",
            # model="/data/njf/Hermes3",
            # model="/data/njf/Astrops",
            messages=[
                {"role": "user", "content": "你是一个专业的语言分析专家。"},
                {"role": "user",
                 "content": "你需要预测餐厅的生存状态（0-倒闭，1"
                            "-存活）（如果评论过于片面或激进时，请参考上述餐厅平均评分进行判断），"
                            "并以评论者的第一视角解释这条评论（解释中不要提及评分和生存预测）。你的输出必须严格遵循格式-'预测: .解释: .'且控制在50字内"},
                {"role": "assistant",
                 "content": "好的，我会解释该评论！我不会生成任何思考过程或无关信息并且使用中文回答"},
                {"role": "user",
                 "content": "这是餐厅的一条评论：" + can +
                            "以及它的平均评分（所有评论的平均分）：" + rate + "注意！不要生成任何无关内容，只能用中文回答"}

            ],
            max_tokens=256,
        )
        import re
        content = completion.choices[0].message.content

        match = re.search(r'预测[：:]\s*(\d+)\s*解释[：:]\s*([^\<]*)', content, re.IGNORECASE)
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
    with open('/home/njf/Rec/data/City3/dataset_evl.pickle', 'rb') as f_check:
        loaded_data = pickle.load(f_check)

    for i in range(0, 2000):
        print(i)
        can = loaded_data[0][i]['input_field_data']['review']
        rate = loaded_data[0][i]['input_field_data']['rating']
        pre = loaded_data[0][i]['input_field_data']['survival']
        res = getOut(can, rate)
        print(can)
        print(res)
        if res != -1:
            ans = calc(can, res['explanation'])
            if all(value == 0 for value in ans.values()):
                continue  # 跳过此循环
            count += 1
            if pre == res['prediction']:
                correct += 1
            ans = calc(can, res['explanation'])
            bleu += ans['BLEU']
            rouge1 += ans['ROUGE-1']
            rouge2 += ans['ROUGE-2']
            rougel += ans['ROUGE-L']
            rougeSu4 += ans['ROUGE-SU4']
            print(ans)
            print('-----------------------------')

except FileNotFoundError as e:
    print(count)
    print(f"Accuracy: {1 - 1.0 * correct / count}")
    print(f"Average BLEU: {bleu / count}")
    print(f"Average ROUGE-1: {rouge1 / count}")
    print(f"Average ROUGE-2: {rouge2 / count}")
    print(f"Average ROUGE-L: {rougel / count}")
    print(f"Average ROUGE-SU4: {rougeSu4 / count}")

# 输出最终结果
print(f"Total count: {count}")
if count > 0:
    print(f"Accuracy: {1 - 1.0 * correct / count}")
    print(f"Average BLEU: {bleu / count}")
    print(f"Average ROUGE-1: {rouge1 / count}")
    print(f"Average ROUGE-2: {rouge2 / count}")
    print(f"Average ROUGE-L: {rougel / count}")
    print(f"Average ROUGE-SU4: {rougeSu4 / count}")
else:
    print("No valid predictions were made.")
