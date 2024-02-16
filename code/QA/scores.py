import json
from sklearn.metrics import accuracy_score

"""
File Paths
"""
positve_true_qa  = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/source/exp-data-pos.json"
negative_true_qa = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/source/exp-data-neg.json"

big_bert_neg_pred    = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/final_syn_bert_big_neg_pred.json"
big_bert_pos_pred    = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/final_syn_bert_big_pos_pred.json"
small_bert_neg_pred  = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/final_syn_bert_small_neg_pred.json"
small_bert_pos_pred  = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/final_syn_bert_small_pos_pred.json"

big_roberta_neg_pred   = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/final_syn_roberta_big_neg_pred.json"
big_roberta_pos_pred   = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/final_syn_roberta_big_pos_pred.json"
small_roberta_neg_pred = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/final_syn_roberta_small_neg_pred.json"
small_roberta_pos_pred = "/Users/sbhar/Riju/PhDCode/ACL/data/QA/final_syn_roberta_small_pos_pred.json"


def read_true_labels(source_labels_path):
        with open(source_labels_path,'r') as file:
                contents = json.load(file)

        data = contents['data']
        id2label = {}
        for key in data:
                element = key
                id      = element['id']
                answers = element['answers']
                for answer in answers:
                        label = answer['input_text']
                id2label[id] = label

        return id2label

def read_pred_labels(pred_labels_path):
        with open(pred_labels_path,'r') as file:
                preds = json.load(file)

        id2labels = {}
        for pred in preds:
                id    = pred['id']
                label = pred['answer']
                id2labels[id] = label

        return id2labels


def calculate_scores(true_file,pred_file,model='Bert',size='large',polarity='pos'):
        true_dict = read_true_labels(true_file)
        pred_dict = read_pred_labels(pred_file)

        true_labels = list(true_dict.values())
        pred_labels = list(pred_dict.values())

        assert len(true_labels) == len(pred_labels)

        if model == 'Bert' and size == 'large' and polarity == 'neg':
                print(f"The Negative Accuracy for Bert large: {accuracy_score(true_labels,pred_labels)}")

        if model == 'Bert' and size == 'large' and polarity == 'pos':
                print(f"The Positive Accuracy for Bert large: {accuracy_score(true_labels,pred_labels)}")

        if model == 'Bert' and size == 'small' and polarity == 'neg':
                print(f"The Negative Accuracy for Bert Small: {accuracy_score(true_labels,pred_labels)}")

        if model == 'Bert' and size == 'small' and polarity == 'pos':
                print(f"The Positive Accuracy for Bert Small: {accuracy_score(true_labels,pred_labels)}")

        if model == 'RoBerta' and size == 'large' and polarity == 'neg':
                print(f"The Negative Accuracy for RoBerta large: {accuracy_score(true_labels,pred_labels)}")

        if model == 'RoBerta' and size == 'large' and polarity == 'pos':
                print(f"The Positive Accuracy for RoBerta large: {accuracy_score(true_labels,pred_labels)}")

        if model == 'RoBerta' and size == 'small' and polarity == 'neg':
                print(f"The Negative Accuracy for RoBerta Small: {accuracy_score(true_labels,pred_labels)}")

        if model == 'RoBerta' and size == 'small' and polarity == 'pos':
                print(f"The Positive Accuracy for RoBerta Small: {accuracy_score(true_labels,pred_labels)}")


if __name__ == "__main__":
        print("------------------------------------------------------------------------------------------")
        print("QA scores for the CoQA models with SYN finetuning")
        print("------------------------------------------------------------------------------------------")
        calculate_scores(
                true_file=negative_true_qa,
                pred_file=big_bert_neg_pred,
                model='Bert',
                size='large',
                polarity='neg'
        )

        calculate_scores(
                true_file=positve_true_qa,
                pred_file=big_bert_pos_pred,
                model='Bert',
                size='large',
                polarity='pos'
        )

        calculate_scores(
                true_file=negative_true_qa,
                pred_file=small_bert_neg_pred,
                model='Bert',
                size='small',
                polarity='neg'
        )

        calculate_scores(
                true_file=positve_true_qa,
                pred_file=small_bert_pos_pred,
                model='Bert',
                size='small',
                polarity='pos'
        )

        print("------------------------------------------------------------------------------------------")
        calculate_scores(
                true_file=negative_true_qa,
                pred_file=big_roberta_neg_pred,
                model='RoBerta',
                size='large',
                polarity='neg'
        )

        calculate_scores(
                true_file=positve_true_qa,
                pred_file=big_roberta_pos_pred,
                model='RoBerta',
                size='large',
                polarity='pos'
        )


        calculate_scores(
                true_file=negative_true_qa,
                pred_file=small_roberta_neg_pred,
                model='RoBerta',
                size='small',
                polarity='neg'
        )

        calculate_scores(
                true_file=positve_true_qa,
                pred_file=small_roberta_pos_pred,
                model='RoBerta',
                size='small',
                polarity='pos'
        )
        print("------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------")





