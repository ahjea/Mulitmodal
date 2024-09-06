import numpy as np
import torch, os, gc, statistics
import models, data_loader
from sklearn.model_selection import KFold


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    fold = 5
    acc_list = []
    f1_list = []

    # TODO set data_path
    data_path = "/home/iis/jemin/multimodal/data/contents/"
    inputs_dataset_content = np.load(data_path + 'input_ids.npy')
    masks_dataset_content = np.load(data_path + 'attentions.npy')
    labels_dataset_content = np.load(data_path + 'input_labels.npy')
    # print(inputs_dataset_content.shape, masks_dataset_content.shape, labels_dataset_content.shape)



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

        # print(len(train_inputs_content), len(val_inputs_content), len(test_inputs_content))


        # def __init__(self, input_ids1, attention_masks1, labels1, input_ids2, attention_masks2, labels2):
        train_dataloader = data_loader.create_content_dataloader(train_inputs_content, train_attention_mask_content, train_labels_content,True, 16)
        validation_dataloader = data_loader.create_content_dataloader(val_inputs_content, val_attention_mask_content, val_labels_content, True, 16)
        prediction_dataloader = data_loader.create_content_dataloader(test_inputs_content, test_attention_mask_content, test_labels_content, False, 16)


        # TODO : change path
        path = "/home/iis/jemin/github/models"
        acc, f1 = models.content_model_test(path, prediction_dataloader, device, i+1)
        acc_list.append(acc)
        f1_list.append(f1)

    print("Model Accuracy is " + str(statistics.median(acc_list)))
    print("Model F1_Score is " + str(statistics.median(f1_list)))


if __name__ == "__main__":
    main()


