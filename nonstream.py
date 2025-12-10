import requests
import json
from datetime import datetime
import os
import pandas as pd
from pathlib import Path

choices = ["A", "B", "C", "D"]
categories = {}
subcategories = {}

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {df.iloc[idx, j + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt

def gen_prompt(train_df, subject, k=5):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k == 0:
        return prompt
    k = min(k, train_df.shape[0])
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

# ---------------------- 非流式API调用函数 ----------------------
def call_api_and_save(mmlu_item, api_url, model_name="default-model", save_dir="mmlu_api_results"):
    subject = mmlu_item["subject"]
    save_dir = Path(save_dir) / subject
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        prompt = mmlu_item["prompt"]
        payload = {
            #"messages": [{"role": "user", "content": prompt}],
            #"temperature": 0.5,
           # "recreate": True,
           # "max_tokens": 4800,
            "model": "qwen3",  # 模型名称可以随意写，但最好和实际模型匹配
            "messages": [
                {"role": "system", "content": "你是一位乐于助人的AI助手。"},
                {"role": "user", "content": prompt + "/no_think"}
            ],
            "temperature": 0.5,
            "max_tokens": 8192,
            "stream": False,
            "cache_prompt": False,
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # 直接解析完整JSON
        try:
            # 非流式请求（无stream参数）
            response = requests.post(
                #url=api_url,
                #headers=headers,
                #json=payload,
                #timeout=800
                API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            data = response.json()
            # 兼容OpenAI格式，或根据你的llama-server实际返回格式调整
            output = data.get("choices", [])[0].get("message", {}).get("content", "")
        except Exception as exception:
            print(exception)
            raise

        result = {
            "id": mmlu_item["id"],
            "subject": subject,
            "question": mmlu_item["question"],
            "options": mmlu_item["options"],
            "prompt": prompt,
            "data": data,
            "model_raw_output": output,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }

        save_path = save_dir / f"{mmlu_item['id']}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"处理完成: {mmlu_item['id']} | 输出预览：{output[:100]}...")
        return output

    except requests.exceptions.RequestException as e:
        error_msg = f"ID: {mmlu_item['id']} 调用失败: {str(e)}\n"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f"服务器返回: {e.response.text}"

        error_dir = Path(save_dir) / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / f"error_{mmlu_item['id']}.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_msg)

        print(f"错误: {mmlu_item['id']} 已保存错误日志")
        return None

def load_mmlu_task(data_dir, subject, ntrain=5):
    dev_path = os.path.join(data_dir, "dev", f"{subject}_dev.csv")
    test_path = os.path.join(data_dir, "test", f"{subject}_test.csv")
    try:
        dev_df = pd.read_csv(dev_path, header=None)[:ntrain]
        test_df = pd.read_csv(test_path, header=None)
        print(f"\n成功加载任务：{subject} | 少样本示例数：{len(dev_df)} | 测试题目数：{len(test_df)}")
        return dev_df, test_df
    except FileNotFoundError as e:
        print(f"错误：未找到任务 {subject} 的文件 - {e}")
        return None, None

def batch_evaluate_mmlu(api_url, data_dir, save_dir="mmlu_api_results", ntrain=5, subjects=None):
    if subjects is None:
        test_dir = os.path.join(data_dir, "test")
        subjects = sorted([
            f.split("_test.csv")[0] for f in os.listdir(test_dir) if "_test.csv" in f
        ])
    print(f"待评估任务数：{len(subjects)} | 任务列表：{subjects}")

    start_time = datetime.now()
    for subject in subjects:
        dev_df, test_df = load_mmlu_task(data_dir, subject, ntrain)
        if dev_df is None or test_df is None:
            continue

       # prompt_prefix = gen_prompt(dev_df, subject, ntrain)
        task_total = len(test_df)
        for idx in range(test_df.shape[0]):
            question = test_df.iloc[idx, 0]
            options = {
                "A": test_df.iloc[idx, 1],
                "B": test_df.iloc[idx, 2],
                "C": test_df.iloc[idx, 3],
                "D": test_df.iloc[idx, 4]
            }
            prompt_suffix = format_example(test_df, idx, include_answer=False)
            #full_prompt = prompt_prefix + prompt_suffix
            full_prompt = prompt_suffix
            mmlu_item = {
                "id": f"{subject}_test_{idx}",
                "subject": subject,
                "question": question,
                "options": options,
                "prompt": full_prompt
            }
            call_api_and_save(mmlu_item, api_url, save_dir=save_dir)

        print(f"\n【任务 {subject} 处理完成】共处理 {task_total} 道题目")

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print("\n" + "=" * 50)
    print(f"所有任务处理完成 | 总耗时：{total_time:.2f}秒 | 处理任务数：{len(subjects)}")

    stats = {
        "total_tasks": len(subjects),
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "cost_time_seconds": total_time,
        "processed_tasks": subjects
    }
    stats_path = Path(save_dir) / "overall_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n整体统计结果保存至：{stats_path}")

if __name__ == "__main__":
    API_URL = "http://127.0.0.1:8080/v1/chat/completions"
    MMLU_DATA_DIR = r"C:\llama\mmlu\mmlu\data\data"
    SAVE_DIR = "./mmlu_api_results_1"
    NTRAIN = 5
    TARGET_SUBJECTS = None

    batch_evaluate_mmlu(
        api_url=API_URL,
        data_dir=MMLU_DATA_DIR,
        save_dir=SAVE_DIR,
        ntrain=NTRAIN,
        subjects=TARGET_SUBJECTS
    )
