from src.config import RERANKED_DATA_PATH

BATCH_OUTPUT_FILE = RERANKED_DATA_PATH / 'batch_output_gpt-5_v02.jsonl'

def get_total_tokens_from_batch_output(filepath):
    """
    Reads the batch output JSONL file and sums up the total tokens used across all responses.
    """
    import json

    total_input_tokens = 0
    total_output_tokens = 0
    user_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                response = data['response']
                body = response['body']
                usage = body.get('usage', {})
                total_input_tokens += usage.get('prompt_tokens')
                total_output_tokens += int(usage.get('completion_tokens'))
                user_count += 1
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in batch output file at {filepath}")
        return 0
    except FileNotFoundError:
        print(f"Error: Batch output file not found at {filepath}")
        return 0
    total_tokens = total_input_tokens + total_output_tokens
    average_input_tokens_per_user = total_input_tokens / user_count
    average_output_tokens_per_user = total_output_tokens / user_count

    return total_input_tokens, total_output_tokens, average_input_tokens_per_user, average_output_tokens_per_user

if __name__ == "__main__":
    input_tokens, output_tokens, avg_inp, avg_outp = get_total_tokens_from_batch_output(BATCH_OUTPUT_FILE)
    print(f"Total Input Tokens: {input_tokens}")
    print(f"Total Output Tokens: {output_tokens}")
    print(f"Average Input Tokens per User: {avg_inp}")
    print(f"Average Output Tokens per User: {avg_outp}")