from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from pre_process_ip import Extract_Features
from coqa_processor import Processor
from coqa_classes import Result
from bert_model import BertBaseUncasedModel
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm
import os
from metrics import get_predictions

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()

output_path      = "/users/melodi/sbhar/NegationExps/CoQA/output/"
predict_file     = "final_syn_bert_small_pos_pred.json"
#eval_model_path  = "/users/melodi/sbhar/NegationExps/CoQA/models/bigbert_coqa.bin"
eval_model_path  = "/users/melodi/sbhar/NegationExps/CoQA/models/smallbert_coqa.bin"


processor = Processor()

#check if mps is available
"""

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        device = torch.device("cpu")
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        device = torch.device("cpu")
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
"""

device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#config = RobertaConfig.from_pretrained('roberta-large')
#config = BertConfig.from_pretrained('bert-large-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')

qa_model = BertBaseUncasedModel(config)
qa_model.load_state_dict(torch.load(eval_model_path,map_location=device), strict=True)
model = qa_model.to(device)
print("Model has been loaded successfully !!")
#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased',do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

print("Tokenizer has been loaded !!")

def main():
        examples = processor.get_examples("/users/melodi/sbhar/NegationExps/CoQA/data",0,filename="exp-data-pos.json",threads=12,dataset_type=None)
        print("The examples has been processed !!")
        features, dataset = Extract_Features(examples=examples,tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=False, threads=12)
        print("The features have been processed as well !!")

        evalutation_sampler = SequentialSampler(dataset)
        evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=1)

        mod_results = []

        for batch in tqdm(evaluation_dataloader, desc="Evaluating"):
            qa_model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],"segment_ids": batch[1],"input_masks": batch[2]}
                example_indices = batch[3]
                outputs = qa_model(**inputs)
            
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                #print("The Unique Id getting passed is: ",unique_id)
                output = [convert_to_list(output[i]) for output in outputs]
                start_logits, end_logits, yes_logits, no_logits, unk_logits = output

                #print("The yes_logits are : ",yes_logits)
                #print("The no logits are : ",no_logits)

                result = Result(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits, yes_logits=yes_logits, no_logits=no_logits, unk_logits=unk_logits)
                mod_results.append(result)
        
        output_prediction_file = os.path.join(output_path, predict_file)
        get_predictions(examples, features, mod_results, 20, 30, True, output_prediction_file, False, tokenizer)
        






if __name__ == '__main__':
    main()



