from sre_constants import IN_UNI_IGNORE
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import json
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', type=str, default="model/gpt2_medium_fine_tuned_coder",
                        help='the path to load fine-tuned model')
    parser.add_argument('--max_length', type=int, default=128,
                        help='maximum length for code generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='temperature for sampling-based code geneeration')
    parser.add_argument(
        "--use_cuda", action="store_true", help="inference with gpu?"
    )

    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    model.eval()
    if args.use_cuda:
        model.to("cuda")

    true_positives = 0  
  
    with open("dataset/test_code/json/test_code.jsonl") as file:
        Lines = file.readlines()
        for line in Lines:
            input_ids = json.loads(line)
            input_ids = input_ids['token_ids']

            context = tokenizer.decode(input_ids, skip_special_tokens=True)

            input_ids = tokenizer.encode("<python> " + context,
                        return_tensors='pt')
            

            for i in range(len(input_ids)):
                outputs = model.generate(input_ids=torch.tensor(input_ids[0:i]) if args.use_cuda else input_ids,
                                                max_length=args.max_length,
                                                temperature=args.temperature,
                                                num_return_sequences=1)
                # print(input_ids, outputs)
                print(outputs)
                print(input_ids[0:i])




                if input_ids == outputs:
                    true_positives = true_positives + 1

                break
                # for j in range(len(input_ids[i:i+1])):
                #     if input_ids[j] == outputs[j]:
                #         true_positives = true_positives + 1
                #     else:
                #         break
                # break


    print(true_positives - 1)
