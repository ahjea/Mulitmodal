import numpy as np
import torch, os, gc
import models, data_loader
from sklearn.model_selection import KFold
from transformers import BertConfig


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    fold = 5
    All_train_loss = []
    All_val_loss = []
    All_ACC = []
    All_test_ACC = []
    All_f1 = []
    All_precision = []
    All_recall = []
    All_confusion = []


    inputs_dataset_content = np.load('/home/iis/jemin/multimodal/data/contents/input_ids.npy')
    masks_dataset_content = np.load('/home/iis/jemin/multimodal/data/contents/attentions.npy')
    labels_dataset_content = np.load('/home/iis/jemin/multimodal/data/contents/input_labels.npy')
    print(inputs_dataset_content.shape, masks_dataset_content.shape, labels_dataset_content.shape)



    kfold = KFold(n_splits=fold, shuffle=False)
    splits = list(kfold.split(inputs_dataset_content))
    split_number = len(inputs_dataset_content)//fold

    for i, (train_idx, test_idx) in enumerate(splits):
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Fold {i + 1}")
        total_train_inputs_content = [inputs_dataset_content[j] for j in train_idx]
        total_train_attention_mask_content = [masks_dataset_content[j] for j in train_idx]
        total_train_labels_content = [labels_dataset_content[j] for j in train_idx]

        test_inputs_content = [inputs_dataset_content[j] for j in test_idx]
        test_attention_mask_content = [masks_dataset_content[j] for j in test_idx]
        test_labels_content = [labels_dataset_content[j] for j in test_idx]


        train_inputs_content = total_train_inputs_content[split_number:]
        train_attention_mask_content = total_train_attention_mask_content[split_number:]
        train_labels_content = total_train_labels_content[split_number:]
        val_inputs_content = total_train_inputs_content[:split_number]
        val_attention_mask_content = total_train_attention_mask_content[:split_number]
        val_labels_content = total_train_labels_content[:split_number]

        print(len(train_inputs_content), len(val_inputs_content), len(test_inputs_content))


        # def __init__(self, input_ids1, attention_masks1, labels1, input_ids2, attention_masks2, labels2):
        train_dataloader = data_loader.create_content_dataloader(train_inputs_content, train_attention_mask_content, train_labels_content,True, 16)
        validation_dataloader = data_loader.create_content_dataloader(val_inputs_content, val_attention_mask_content, val_labels_content, True, 16)
        prediction_dataloader = data_loader.create_content_dataloader(test_inputs_content, test_attention_mask_content, test_labels_content, False, 16)


        train_loss_history, val_loss_history, acc_history, test_acc, f1, recall, precision, confusion = models.content_model_train(train_dataloader, validation_dataloader, prediction_dataloader, 100, device, i+1)
        print("====================Train Result====================")
        print(f"Fold {i + 1} Train_loss_history : ")
        for tarin_loss in train_loss_history:

            print(tarin_loss)
        print(f"Fold {i + 1} Val_loss_history : ")
        for val_loss in val_loss_history:
            print(val_loss)
        print(f"Fold {i + 1} ACC_history : ", acc_history)
        for acc in acc_history:
            print(acc)
        print(f"Fold {i + 1} Testset_ACC : ", test_acc)

        All_train_loss.append(train_loss_history)
        All_val_loss.append(val_loss_history)
        All_ACC.append(acc_history)
        All_test_ACC.append(test_acc)
        All_f1.append(f1)
        All_recall.append(recall)
        All_precision.append(precision)
        All_confusion.append(confusion)


    print("================================Summaries================================")
    for i in range(fold):
        print(i+1, " Step result")
        print("Train loss history")
        for value in All_train_loss[i]:
            print(value)
        print("Validation loss history")
        for value in All_val_loss[i]:
            print(value)
        print("Validation Accuracy")
        for value in All_ACC[i]:
            print(value)
        print("Test set Accuracy : ", All_test_ACC[i])
        print("Test set F1 : ", All_f1[i])
        print("Test set Recall : ", All_recall[i])
        print("Test set Precision : ", All_precision[i])
        print("Test set confusion matrix : ", All_confusion[i])


if __name__ == "__main__":
    main()


