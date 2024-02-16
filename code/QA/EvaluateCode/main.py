import warnings
# Turn off all warnings
warnings.filterwarnings("ignore")


import torch
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig,BertModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pre_process_ip import Extract_Features
from coqa_processor import Processor
from train_model import train
#from train_model_2 import train
from bert_model import BertBaseUncasedModel
#from rb_model2 import RobertaBaseModel
import json
import numpy as np

config_file_path = "/users/melodi/sbhar/NewMaskingTraining/new_exp_bert/config.json"

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()


            

def main():
    def read_config(config_file_path):
        with open(config_file_path) as f:
            data = json.load(f)
    
        train_file            = data["PARAMETERS"]["train_file"]
        predict_file          = data["PARAMETERS"]["predict_file"]
        pretrained_model      = data["PARAMETERS"]["pretrained_model"]
        output_path           = data["PARAMETERS"]["output_path"]
        epochs                = data["PARAMETERS"]["epochs"]
        evaluation_batch_size = data["PARAMETERS"]["evaluation_batch_size"]
        train_batch_size      = data["PARAMETERS"]["train_batch_size"]

        return train_file,predict_file,pretrained_model,output_path,epochs,evaluation_batch_size,train_batch_size
    
    train_file,_,pretrained_model,_,_,_,_ = read_config(config_file_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained(pretrained_model,do_lower_case=True)
    #col_str  = "black blue green red white yellow"
    col_str   = "black blue green red white yellow modest elegant plastic wooden cotton synthetic glass crystal"
    col_inds = tokenizer(col_str)
    col_inds = torch.tensor(col_inds["input_ids"][1:-1])
    
    """
    adj_str  = "modest elegant plastic wooden cotton synthetic glass crystal"
    adj_inds = tokenizer(adj_str)
    adj_inds = torch.tensor(adj_inds["input_ids"][1:-1])
    """

    obj_str  = "house car bat ball shirt coat window door glass table"
    #obj_str  = "house car bat ball shirt jacket window door glass table"
    obj_inds = tokenizer(obj_str)
    obj_inds = torch.tensor(obj_inds["input_ids"][1:-1])
     

    process = Processor()

    
    examples  = process.get_examples("/users/melodi/sbhar/NewMaskingTraining/data",history_len=0,threads=12)
    features, dataset = Extract_Features(examples=examples,
            tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=False, threads=12)
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=1)
    data_l = list(evaluation_dataloader)
    
    sample_feature = features[0]
    print("The input tokens are: ",sample_feature.tokens)
    for i in range(len(examples)):
        if examples[i].qas_id.startswith("4"):
            batch = data_l[i]
            inp   = batch[0][:,:batch[2].sum().item()]
            
            att_mask = torch.ones((1,inp.shape[1]),dtype=torch.int64)
            indices_list = []

            q_obj_ind = torch.where(inp[0]==inp[0][3])[0][1]
            q_col_ind = torch.where(inp[0]==inp[0][4])[0][1]

            for c in col_inds:
                #for a color not presant inside the question but presant inside the input ids
                if c!=inp[0][4].item() and c in inp[0].tolist():
                    nq_col_ind = torch.where(inp[0]==c)[0][0]

            for o in obj_inds:
                if o!=inp[0][3] and o in inp[0].tolist():
                    nq_obj_ind = torch.where(inp[0]==o)[0][0]
            
            assert inp[0][4].item() in col_inds
            assert inp[0][q_col_ind].item() in col_inds
            assert inp[0][nq_col_ind].item() in col_inds
            assert inp[0][3].item() in obj_inds
            assert inp[0][q_obj_ind].item() in obj_inds
            assert inp[0][nq_obj_ind].item() in obj_inds

            
            assert inp[0][q_obj_ind-1].item() in col_inds
            assert inp[0][q_obj_ind-2].item() in col_inds
            assert inp[0][nq_obj_ind-1].item() in col_inds
            assert inp[0][nq_obj_ind-2].item() in col_inds

            indices_list.append(q_obj_ind.item())
            indices_list.append(nq_obj_ind.item())

            att_mask[0,q_obj_ind-1] = 0 #masking the color 
            att_mask[0,q_obj_ind-2] = 0 #masking the adjective

            examples[i].new_mask = att_mask.tolist()[0] 
            examples[i].indices = indices_list

        #print(examples[i].doc_tokens)
        #print("The masking position is: ",q_obj_ind)
        

    #print("===============================")
    #print("Entering the next step")
    #print("===============================")
    for i in range(len(examples)):
        if examples[i].qas_id.startswith("3"):
            batch = data_l[i]
            inp   = batch[0][:,:batch[2].sum().item()]
            att_mask = torch.ones((1,inp.shape[1]),dtype=torch.int64)

            indices_list = []
            print("The question tokens are: ",examples[i].question_text)
            print("The context tokens are:  ",examples[i].doc_tokens)

            q_obj_ind = torch.where(inp[0]==inp[0][5])[0][1]
            q_col_ind = torch.where(inp[0]==inp[0][4])[0][1]
            for c in col_inds:
                if c!=inp[0][4].item() and c in inp[0].tolist():
                    nq_col_ind = torch.where(inp[0]==c)[0][0]
            for o in obj_inds:
                if o!=inp[0][5].item() and o in inp[0].tolist():
                    nq_obj_ind = torch.where(inp[0]==o)[0][0]
            
            assert inp[0][4].item() in col_inds
            assert inp[0][q_col_ind].item() in col_inds
            assert inp[0][nq_col_ind].item() in col_inds
            assert inp[0][5].item() in obj_inds
            assert inp[0][q_obj_ind].item() in obj_inds
            assert inp[0][nq_obj_ind].item() in obj_inds
            
            assert inp[0][q_obj_ind-1].item() in col_inds
            assert inp[0][q_obj_ind-2].item() in col_inds
            assert inp[0][nq_obj_ind-1].item() in col_inds
            assert inp[0][nq_obj_ind-2].item() in col_inds

            indices_list.append(q_obj_ind.item())
            indices_list.append(nq_obj_ind.item())

            
            att_mask[0,q_obj_ind-1] = 0 #masking the color 
            att_mask[0,q_obj_ind-2] = 0 #masking the adjective
            
            examples[i].new_mask = att_mask.tolist()[0]
            examples[i].indices = indices_list
        
        #print(examples[i].doc_tokens)
        #print("The masking position is: ",q_obj_ind)
        
    
    print("Attention Masks calculated !!")
    #calculating the final dataset with the masking positions pre-calculated
    final_features, final_dataset = Extract_Features(examples, tokenizer, max_seq_length=512, doc_stride=128, max_query_length=64, is_training=True, threads=12,incl_mask=True)
    final_feature = final_features[0]
    if len(final_feature.new_mask) == len(final_feature.input_mask):
        print("The shapes of the two masks match!!!")
    else:
        raise ValueError("The Shapes Do not match,Pls Check")
    
    print("The final features have also been calculated!!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained(pretrained_model)

    qa_model = BertBaseUncasedModel(config)

    #loading the pretrained Model Weights
    #qa_model.load_state_dict(torch.load("/users/melodi/sbhar/NewMaskingTraining/model/rb_pytorch_model.bin",map_location=torch.device('cpu')), strict=True)
    qa_model.load_state_dict(torch.load("/users/melodi/sbhar/NewMaskingTraining/model/smallbert_coqa.bin",map_location=device), strict=True)
    #qa_model.load_state_dict(torch.load("/users/melodi/sbhar/NewMaskingTraining/TrainedModelsCoat/Test10GradsOneDummy/model_weights/pytorch_model.bin",map_location=device), strict=True)

    qa_model = qa_model.to(device)

    print("The Model has been Loaded Sucessfully...")

    
    
    
    print("Check a sample mask: ")
    print("The input tokens are: ",final_feature.tokens)
    print("The mask is:",final_feature.new_mask)
        
    train(config_file_path,final_dataset,model=qa_model,tokenizer=tokenizer,device=device)
    print("The model has been trained")

if __name__ == '__main__':
    main()