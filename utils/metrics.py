import copy
import math
from utils.tools import sync_dict, vague_map


def task_register(task):
    metrics_dict = {
        'count': 0,
        'correct': 0,
        'bleu': 0.0,
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougel': 0.0,
        'rougeSu4': 0.0,
        'RewardSum': 0.0,
        'NonExistRate': 0.0,
        'RepeatRate': 0.0,
        'CorrectCount': 0.0,
        'Count': 1e-24,
        'Recall': 0.0,
        'MRR': 0.0,
        'NDCG': 0.0,
        'TargetCategoryRate': 0.0,
        'InHistoryRate': 0.0,
    }
    return metrics_dict


class Metrics:
    def __init__(self, tasks, topk, category2item, title2item, accelerator=None):
        self.tasks = tasks
        self.topk = topk
        self.category2item = category2item
        self.title2item = title2item
        self.accelerator = accelerator

        self.metrics_dict = {_: task_register(_) for _ in self.tasks}

    def add_sample(self, task, input_field_data, output_titles, target_title, list_reward=0.0, vague_mapping=True):
        print(input_field_data)
        print('-----------------')
        print(output_titles)
        print('-----------------')
        print()
        '''
        if output_titles:
            # 获取第一个选择的消息内容
            content = output_titles
            # 使用正则表达式提取第一个 prediction 和对应的 explanation
            cleaned_content = re.sub(r'[^a-zA-Z0-9\s:.]', '', content)

            match = re.search(r'prediction:\s*(\d+)\s*explanation:\s*([^\<]*)', cleaned_content,
                              re.IGNORECASE)
            if match:
                prediction = match.group(1)
                explanation = match.group(2).strip()
                if prediction == " " or explanation == "":
                    return -1
                res = {
                    "prediction": prediction,
                    "explanation": explanation
                }

        if res["prediction"] == input_field_data['prediction']:
            self.metrics_dict[task]['correct'] += 1
        if task == "SurvivalPre":
            ans = cal(input_field_data['explanation'], res["explanation"])
        else:
            ans = cacl(input_field_data['explanation'], res["explanation"])
        self.metrics_dict[task]['BLEU'] += ans['BLEU']
        self.metrics_dict[task]['ROUGE-1'] += ans['ROUGE-1']
        self.metrics_dict[task]['ROUGE-2'] += ans['ROUGE-2']
        self.metrics_dict[task]['ROUGE-L'] += ans['ROUGE-L']
        self.metrics_dict[task]['ROUGE-SU4'] += ans['ROUGE-SU4']
        print(ans)
        print('-----------------------------')
        '''


    def __getitem__(self, item):
        return self.metrics_dict[item]

    def __iter__(self):
        return iter(self.metrics_dict.keys())

    def get_sync_metrics(self):
        """
        get the synchronized metrics dict cross all processes while using multi gpus in training.
        :return:
        """
        temp = copy.deepcopy(self.metrics_dict)
        if self.accelerator:
            temp = {t: sync_dict(self.accelerator, temp[t]) for t in temp}
        return temp

    def print(self, temp=None):
        if self.accelerator and not self.accelerator.is_main_process:
            return
        if temp is None:
            temp = copy.deepcopy(self.metrics_dict)
        temp = {_: {__: f'{temp[_][__] / temp[_]["count"]:.4f}' if __ != 'count' else f'{int(temp[_][__])}' for __ in
                    temp[_]} for _ in temp}
        tasks = [_ for _ in temp]
        metrics = ['count',
                   'correct',
                   'bleu'
                   'rouge1',
                   'rouge2',
                   'rougel',
                   'rougeSu4',
                   'NonExistRate',
                   'RepeatRate',
                   'InHistoryRate',
                   'CorrectCount',
                   'Count',
                   'Recall',
                   'MRR',
                   'NDCG',
                   'TargetCategoryRate',
                   'SRTargetCategoryRate',
                   'CategoryRateCorrect',
                   'SRCategoryRateCorrect',
                   'NotInCandidateRate',
                   'Loss',
                   'RewardSum']
        table_rows = [
            f"|{_.center(24)}|{'|'.join([str(temp[__][_]).center(len(__) + 4) if _ in temp[__] else '/'.center(len(__) + 4) for __ in tasks])}|"
            for _ in metrics]
        table_rows_str = '\n'.join(table_rows)
        print(f'''
-{('tasks' + '@' + str(self.topk)).center(24, '-')}-{'-'.join([_.center(len(_) + 4, '-') for _ in tasks])}-
{table_rows_str}
-{'-' * 24}-{'-'.join(['-' * (len(_) + 4) for _ in tasks])}-''')
