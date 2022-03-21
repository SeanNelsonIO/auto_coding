from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
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

    # load fine-tunned model and tokenizer from path
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    model.eval()
    if args.use_cuda:
        model.to("cuda")

    # now the fine-tunned model supports two programming languages, namely, python and java
    def lang_select():
        lang = ""
        while lang not in ["python", "java"]:
            print('Enter the programming language you prefer (python or java)')
            lang = input(">>> ").lower()
        return lang


    lang = lang_select()

    context = ""
    while context != "exit":
        print(f'You are using {lang} now. Enter the context code (exit or change_lang)')
        context = input(">>> ")

        if context == "change_lang":
            lang = lang_select()

            print(f"You are using {lang} now. Enter the context code")
            context = input(">>> ")

        input_ids = tokenizer.encode("<python> " + context,
                                     return_tensors='pt') if lang == "python" else tokenizer.encode(
            "<java> " + context, return_tensors='pt')
        
        print(input_ids)
        outputs = model.generate(input_ids=input_ids.to("cuda") if args.use_cuda else input_ids,
                                 max_length=args.max_length,
                                 temperature=args.temperature,
                                 num_return_sequences=1)

        beam_outputs = model.generate(
            input_ids, 
            max_length=128, 
            num_beams=5, 
            early_stopping=True,
            num_return_sequences=5
        )
        print(input_ids.shape)
        with torch.no_grad():
            output = model(input_ids)
            logits = output.logits[:, -1, :]
            logits = logits[0]
            print("\nAll logits for next word: ")
            print(logits)
            print(logits.shape)
            # prediction for one at a time
            # for i in range(10):  
            #     pred_id = torch.argmax(logits).item()
            #     print("\nPredicted token ID of next word: ")
            #     print(pred_id)
            #     pred_word = tokenizer.decode(pred_id)
            #     print(pred_word)
            #     #remove this prediction
            #     logits = torch.cat([logits[0:pred_id], logits[pred_id + 1:]])
                
            for i in range(100):
                output = model(input_ids)
                logits = output.logits[:, -1, :]
                pred_id = torch.argmax(logits).item()

                # ADD ELEMENT TO INPUT LIST
                tmp_elements = []
                for element in input_ids[0]:
                    tmp_elements.append(element)
                tmp_elements.append(torch.tensor(pred_id))
                input_ids = torch.tensor([tmp_elements])

            
            print(input_ids)
            output_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print('output Generated {}: {}'.format(i, output_decoded))


            #input file
            #start predicting the next token given all of the past tokens as input??
            #predict based on only previous token??
            #if the prediction is wrong should we continue attempt to predict the next token?
            #yes but replace the wrong token with the correct token as the input for the
            #next prediction
        

            # print("\nPredicted token ID of next word: ")
            # print(pred_id)
            # logits = torch.cat([logits[0:pred_id], logits[pred_id + 1:]])
            # pred_id = torch.argmax(logits).item()



            
            # for i in logits:
            #     print("\nAll logits for next word: ")
            #     print(logits)
            #     print(logits.shape)

            #     pred_id = torch.argmax(logits).item()
            #     print("\nPredicted token ID of next word: ")
            #     print(pred_id)

            #     pred_word = tokenizer.decode(pred_id)
            #     print("\nPredicted next word for sequence: ")
            #     print(pred_word)

            # decoded = tokenizer.decode([output[0]], skip_special_tokens=False)
            # print('Generated {}: {}'.format(i, decoded))
            # for i in logits[0]:
            #     print('this is i:', i)
            #     pred_id = torch.argmax(torch.tensor(i)).item()
            #     print("\nPredicted token ID of next word: ")
            #     print(pred_id)

            #     pred_word = tokenizer.decode(pred_id)
            #     print("\nPredicted next word for sequence: ")
            #     print(pred_word)

        
        # for i in range(1):
        #     print(beam_outputs)
        #     # decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
        #     beam_decoded = tokenizer.decode([beam_outputs[i]], skip_special_tokens=False)
        # #     # ends with occurence of double new lines (to meet the convention of code completion)
        # #     # if "\n\n" in decoded:
        # #     #     decoded = decoded[:decoded.index("\n\n")]

        # #     # if "\n\n" in beam_decoded:
        # #     #     beam_decoded = beam_decoded[:beam_decoded.index("\n\n")]

        # #     # print('Generated {}: {}'.format(i, decoded))

        #     print('beam Generated {}: {}'.format(i, beam_decoded))
