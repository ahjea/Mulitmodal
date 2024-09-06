import numpy as np
import time, tqdm
import torch.nn.functional as F
import torch, datetime, math, gc
import torch.nn as nn
from transformers import BertConfig, BertModel, AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def cal_acc(preds, true_labels):
    ret = []
    length = len(preds)
    if len(preds) != len(true_labels):
        if len(preds) > len(true_labels):
            length = len(true_labels)
    for i in range(length):
        if preds[i] == true_labels[i]:
            ret.append(int(1))
        else:
            ret.append(int(0))
    return ret


class MultiModalModel(nn.Module):
    def __init__(self, bert_model1, bert_model2, device):
        super(MultiModalModel, self).__init__()
        self.device = device
        self.model1 = bert_model1
        self.model2 = bert_model2
        self.fc = nn.Linear(2*512, 768*2)
        self.output_layer = nn.Linear(768*2, 1)


    def forward(self, input11, input12, input13, input21, input22, input23):
        outputs1 = self.model1(input11, attention_mask=input12, output_hidden_states=True)
        outputs2 = self.model2(input21, attention_mask=input22, output_hidden_states=True)

        # (16, 512, 768)
        last_hidden_states1 = outputs1.hidden_states[-1]
        last_hidden_states2 = outputs2.hidden_states[-1]

        # Average Pooling, (16,512)
        # average1 = last_hidden_states1.mean(dim=2)
        # average2 = last_hidden_states1.mean(dim=2)
        median1 = torch.median(last_hidden_states1, dim=2)[0]
        median2 = torch.median(last_hidden_states2, dim=2)[0]

        # (16, 512, 768*2)
        concatenated_hidden_states = torch.cat((median1, median2), dim=1)

        fc_output = self.fc(concatenated_hidden_states)
        fc_relu_output = F.gelu(fc_output)

        output = self.output_layer(fc_relu_output)
        output = F.sigmoid(output)
        return output

def create_model(config):
    model = BertForSequenceClassification(config)
    print(model.config)
    model.cuda()

    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    return model


def url_model_train(train_dataloader, val_dataloader, prediction_dataloader, epochs, device, fold):
    config1 = BertConfig(
        vocab_size=1000,
        output_attentions=True,
        output_hidden_states=True,
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = create_model(config1)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(768, 2),
        torch.nn.Sigmoid()
    )
    print(model.classifier)
    model.to(device)

    learning_rate = 1e-5
    print(" Learning rate : ", learning_rate)
    minimum_val_loss = 100
    train_loss_history = np.array([])
    val_loss_history = np.array([])
    acc_history = np.array([])
    patience = 5
    patience_counter = 0

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        model.train()
        train_loss = 0
        t0 = time.time()
        for step, batch in enumerate(train_dataloader):
            if step % 200 == 0 and not step == 0:
                print((
                          f'Step / Total : {step} / {len(train_dataloader)} | Elapse time : {format_time(time.time() - t0)}'))
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
            loss = criterion(outputs.logits, labels_one_hot)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_loss = train_loss / len(train_dataloader)
        train_loss_history = np.append(train_loss_history, train_loss)
        print(train_loss_history.tolist())
        print(f'Train Loss: {train_loss:.4f}')
        print("Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ==================================================================
        #                           Validation
        # ==================================================================
        print("Running Validation...")
        preds = np.array([])
        true_labels = np.array([])
        t0 = time.time()
        model.eval()
        eval_loss = 0
        val_acc = 0
        nb_eval_steps = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
                loss = criterion(outputs.logits, labels_one_hot)
                logits = outputs[1]
                eval_loss += loss.item()
                # eval_loss += loss.item()
                nb_eval_steps += 1
                preds = np.append(preds, logits.to('cpu').numpy())
                true_labels = np.append(true_labels, labels.to('cpu').numpy())
        preds = preds.reshape(len(true_labels), 2)
        preds = np.argmax(preds, axis=1).flatten()
        compare = cal_acc(preds, true_labels)
        val_acc = sum(compare) / len(compare)
        eval_loss = eval_loss / nb_eval_steps
        if eval_loss < minimum_val_loss:
            minimum_val_loss = eval_loss
            torch.save(model, '/home/iis/jemin/github/models/url_model' + str(fold) + '.pt')
            print("model saved")
        val_loss_history = np.append(val_loss_history, eval_loss)
        acc_history = np.append(acc_history, val_acc)
        print(val_loss_history.tolist())
        print(acc_history.tolist())
        print(f'Val Loss: {eval_loss:.4f} | Accuracy: {val_acc:.4f}')
        print("Validation took: {:}".format(format_time(time.time() - t0)))
        print("Training Complete")
        if minimum_val_loss >= eval_loss:
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1
        if patience_counter >= patience:
            print("Model start to overfit. Training Stopped")
            break

    # ==================================================================
    #                           Test
    # ==================================================================
    test_preds_labels_list = []
    test_true_labels_list = []

    model = torch.load('/home/iis/jemin/github/models/url_model' + str(fold) + '.pt', map_location=device)
    model.eval()
    model.to(device)

    for test_batch in prediction_dataloader:
        test_input_ids = test_batch[0].to(device)
        test_attention_mask = test_batch[1].to(device)
        test_labels = test_batch[2].to(device)
        with torch.no_grad():
            outputs = model(test_input_ids, attention_mask=test_attention_mask)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)
        test_preds_labels_list.extend(predicted_labels.to('cpu').numpy())
        test_true_labels_list.extend(test_labels.to('cpu').numpy())

    test_preds_labels = np.array(test_preds_labels_list)
    test_true_labels = np.array(test_true_labels_list)
    compare = cal_acc(test_preds_labels, test_true_labels)
    test_acc = sum(compare) / len(compare)
    print(f'Test dataset Accuracy : {test_acc:.4f}')

    f1 = f1_score(test_true_labels, test_preds_labels, average='weighted')
    recall = recall_score(test_true_labels, test_preds_labels, average='weighted')
    precision = precision_score(test_true_labels, test_preds_labels, average='weighted')
    auc = roc_auc_score(test_true_labels, test_preds_labels, average='weighted', )
    confusion = confusion_matrix(test_true_labels, test_preds_labels)

    print('F1_score : {: .10f}'.format(f1))
    print('Recall : {: .10f}'.format(recall))
    print('Precision : {: .10f}'.format(precision))
    print('AUC : {: .10f}'.format(auc))
    print(confusion)

    # performance metrics : AUC / precision / recall / F1 score /

    return train_loss_history, val_loss_history, acc_history, test_acc, f1, recall, precision, confusion




def content_model_train(train_dataloader, val_dataloader, prediction_dataloader, epochs, device, fold):
    config1 = BertConfig(
        output_attentions=True,
        output_hidden_states=True,
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = create_model(config1)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(768, 2),
        torch.nn.Sigmoid()
    )
    print(model.classifier)
    model.to(device)

    learning_rate = 1e-5
    print(" Learning rate : ", learning_rate)
    minimum_val_loss = 100
    train_loss_history = np.array([])
    val_loss_history = np.array([])
    acc_history = np.array([])
    patience = 5
    patience_counter = 0

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        model.train()
        train_loss = 0
        t0 = time.time()
        for step, batch in enumerate(train_dataloader):
            if step % 200 == 0 and not step == 0:
                print((
                          f'Step / Total : {step} / {len(train_dataloader)} | Elapse time : {format_time(time.time() - t0)}'))
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
            loss = criterion(outputs.logits, labels_one_hot)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_loss = train_loss / len(train_dataloader)
        train_loss_history = np.append(train_loss_history, train_loss)
        print(train_loss_history.tolist())
        print(f'Train Loss: {train_loss:.4f}')
        print("Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ==================================================================
        #                           Validation
        # ==================================================================
        print("Running Validation...")
        preds = np.array([])
        true_labels = np.array([])
        t0 = time.time()
        model.eval()
        eval_loss = 0
        val_acc = 0
        nb_eval_steps = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
                loss = criterion(outputs.logits, labels_one_hot)
                logits = outputs[1]
                eval_loss += loss.item()
                # eval_loss += loss.item()
                nb_eval_steps += 1
                preds = np.append(preds, logits.to('cpu').numpy())
                true_labels = np.append(true_labels, labels.to('cpu').numpy())
        preds = preds.reshape(len(true_labels), 2)
        preds = np.argmax(preds, axis=1).flatten()
        compare = cal_acc(preds, true_labels)
        val_acc = sum(compare) / len(compare)
        eval_loss = eval_loss / nb_eval_steps
        if eval_loss < minimum_val_loss:
            minimum_val_loss = eval_loss
            torch.save(model, '/home/iis/jemin/github/models/content_model' + str(fold) + '.pt')
            print("model saved")
        val_loss_history = np.append(val_loss_history, eval_loss)
        acc_history = np.append(acc_history, val_acc)
        print(val_loss_history.tolist())
        print(acc_history.tolist())
        print(f'Val Loss: {eval_loss:.4f} | Accuracy: {val_acc:.4f}')
        print("Validation took: {:}".format(format_time(time.time() - t0)))
        print("Training Complete")
        if minimum_val_loss >= eval_loss:
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1
        if patience_counter >= patience:
            print("Model start to overfit. Training Stopped")
            break

    # ==================================================================
    #                           Test
    # ==================================================================
    test_preds_labels_list = []
    test_true_labels_list = []

    model = torch.load('/home/iis/jemin/github/models/content_model' + str(fold) + '.pt', map_location=device)
    model.eval()
    model.to(device)

    for test_batch in prediction_dataloader:
        test_input_ids = test_batch[0].to(device)
        test_attention_mask = test_batch[1].to(device)
        test_labels = test_batch[2].to(device)
        with torch.no_grad():
            outputs = model(test_input_ids, attention_mask=test_attention_mask)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)
        test_preds_labels_list.extend(predicted_labels.to('cpu').numpy())
        test_true_labels_list.extend(test_labels.to('cpu').numpy())

    test_preds_labels = np.array(test_preds_labels_list)
    test_true_labels = np.array(test_true_labels_list)
    compare = cal_acc(test_preds_labels, test_true_labels)
    test_acc = sum(compare) / len(compare)
    print(f'Test dataset Accuracy : {test_acc:.4f}')

    f1 = f1_score(test_true_labels, test_preds_labels, average='weighted')
    recall = recall_score(test_true_labels, test_preds_labels, average='weighted')
    precision = precision_score(test_true_labels, test_preds_labels, average='weighted')
    auc = roc_auc_score(test_true_labels, test_preds_labels, average='weighted', )
    confusion = confusion_matrix(test_true_labels, test_preds_labels)

    print('F1_score : {: .10f}'.format(f1))
    print('Recall : {: .10f}'.format(recall))
    print('Precision : {: .10f}'.format(precision))
    print('AUC : {: .10f}'.format(auc))
    print(confusion)

    # performance metrics : AUC / precision / recall / F1 score /

    return train_loss_history, val_loss_history, acc_history, test_acc, f1, recall, precision, confusion


def dual_model_train(train_dataloader, val_dataloader, prediction_dataloader, epochs, device, fold):

    config1 = BertConfig(
        vocab_size=200,
        output_attentions=True,
        output_hidden_states=True,
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model1 = create_model(config1)

    config2 = BertConfig(
        output_attentions=True,
        output_hidden_states=True,
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model2 = create_model(config2)

    patience_counter = 0
    patience = 5
    learning_rate = 1e-5
    print(" Learning rate : ", learning_rate)
    minimum_val_loss = 100
    train_loss_history = np.array([])
    val_loss_history = np.array([])
    acc_history = np.array([])

    model = MultiModalModel(model1, model2, device)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        model.train()
        train_loss = 0
        t1 = time.time()

        for step, batch in enumerate(train_dataloader):
            if step % 200 == 0 and not step == 0:
                print((f'Step / Total : {step} / {len(train_dataloader)} | Elapse time : {format_time(time.time() - t1)}'))
            input_ids_url = batch[0].to(device)
            attention_mask_url = batch[1].to(device)
            labels_url = batch[2].to(device)
            input_ids_content = batch[3].to(device)
            attention_mask_content = batch[4].to(device)
            labels_content = batch[5].to(device)

            outputs = model(input_ids_url, attention_mask_url, labels_url, input_ids_content, attention_mask_content, labels_content)

            labels = labels_url.unsqueeze(1).float()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        train_loss_history = np.append(train_loss_history, train_loss)
        print(train_loss_history.tolist())
        print(f'Train Loss: {train_loss:.4f}')
        print("Training epoch took: {:}".format(format_time(time.time() - t1)))

        ########################## Validation #################################
        print("Running Validation...")
        preds = np.array([])
        true_labels = np.array([])
        t0 = time.time()
        model.eval()
        eval_loss = 0
        val_acc = 0
        nb_eval_steps = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids_url = batch[0].to(device)
                attention_mask_url = batch[1].to(device)
                labels_url = batch[2].to(device)
                input_ids_content = batch[3].to(device)
                attention_mask_content = batch[4].to(device)
                labels_content = batch[5].to(device)
                outputs = model(input_ids_url, attention_mask_url, labels_url, input_ids_content,
                                attention_mask_content, labels_content)
                labels = labels_url.unsqueeze(1).float()
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                nb_eval_steps += 1
                pred_labels = (outputs >= 0.5).float()
                # pred_labels = torch.argmax(outputs, dim=-1)
                preds = np.append(preds, pred_labels.to('cpu').numpy())
                true_labels = np.append(true_labels, labels.to('cpu').numpy())
        # preds = preds.reshape(len(true_labels), 2)
        # preds = np.argmax(preds, axis=1).flatten()
        compare = cal_acc(preds, true_labels)
        val_acc = sum(compare) / len(compare)
        eval_loss = eval_loss / nb_eval_steps
        if eval_loss < minimum_val_loss:
            minimum_val_loss = eval_loss
            torch.save(model,
                       '/home/iis/jemin/github/models/multimodal' + str(fold) + '.pt')
            print("model saved")
            patience_counter = 0
        else:
            patience_counter += 1
        val_loss_history = np.append(val_loss_history, eval_loss)
        acc_history = np.append(acc_history, val_acc)
        print(val_loss_history.tolist())
        print(acc_history.tolist())
        print(f'Val Loss: {eval_loss:.4f} | Accuracy: {val_acc:.4f}')
        print("Validation took: {:}".format(format_time(time.time() - t0)))
        print("Training Complete")
        if patience_counter == patience:
            print("Model start to overfit. Training Stopped")
            break

    ########################## TEST #################################
    test_preds_labels_list = np.array([])
    test_true_labels_list = np.array([])
    model = torch.load('/home/iis/jemin/github/models/multimodal' + str(fold) + '.pt',
                       map_location=device)
    model.eval()
    model.to(device)
    for test_batch in prediction_dataloader:
        input_ids_url = test_batch[0].to(device)
        attention_mask_url = test_batch[1].to(device)
        labels_url = test_batch[2].to(device)
        input_ids_content = test_batch[3].to(device)
        attention_mask_content = test_batch[4].to(device)
        labels_content = test_batch[5].to(device)
        with torch.no_grad():
            outputs = model(input_ids_url, attention_mask_url, labels_url, input_ids_content,
                        attention_mask_content, labels_content)
        pred_labels = (outputs >= 0.5).float()
        labels = labels_url.unsqueeze(1).float()
        test_preds_labels_list = np.append(test_preds_labels_list, pred_labels.to('cpu').numpy())
        test_true_labels_list = np.append(test_true_labels_list, labels.to('cpu').numpy())
        # _, predicted_labels = torch.max(outputs, dim=1)
        # test_preds_labels_list.extend(predicted_labels.to('cpu').numpy())
        # test_true_labels_list.extend(labels_url.to('cpu').numpy())
    test_preds_labels = np.array(test_preds_labels_list)
    test_true_labels = np.array(test_true_labels_list)
    compare = cal_acc(test_preds_labels, test_true_labels)
    test_acc = sum(compare) / len(compare)
    print(f'Test dataset Accuracy : {test_acc:.4f}')

    f1 = f1_score(test_true_labels, test_preds_labels, average='weighted')
    recall = recall_score(test_true_labels, test_preds_labels, average='weighted')
    precision = precision_score(test_true_labels, test_preds_labels, average='weighted')
    confusion = confusion_matrix(test_true_labels, test_preds_labels)

    print('F1_score : {: .10f}'.format(f1))
    print('Recall : {: .10f}'.format(recall))
    print('Precision : {: .10f}'.format(precision))
    print(confusion)

    return train_loss_history, val_loss_history, acc_history, test_acc, f1, recall, precision, confusion



def content_model_test(path, prediction_dataloader, device, fold):
    test_preds_labels_list = []
    test_true_labels_list = []

    model = torch.load(path + '/content_model' + str(fold) + '.pt', map_location=device)
    model.eval()
    model.to(device)

    for test_batch in prediction_dataloader:
        test_input_ids = test_batch[0].to(device)
        test_attention_mask = test_batch[1].to(device)
        test_labels = test_batch[2].to(device)
        with torch.no_grad():
            outputs = model(test_input_ids, attention_mask=test_attention_mask)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)
        test_preds_labels_list.extend(predicted_labels.to('cpu').numpy())
        test_true_labels_list.extend(test_labels.to('cpu').numpy())

    test_preds_labels = np.array(test_preds_labels_list)
    test_true_labels = np.array(test_true_labels_list)
    compare = cal_acc(test_preds_labels, test_true_labels)
    test_acc = sum(compare) / len(compare)
    # print(f'Test dataset Accuracy : {test_acc:.4f}')

    f1 = f1_score(test_true_labels, test_preds_labels, average='weighted')
    # recall = recall_score(test_true_labels, test_preds_labels, average='weighted')
    # precision = precision_score(test_true_labels, test_preds_labels, average='weighted')
    # auc = roc_auc_score(test_true_labels, test_preds_labels, average='weighted', )
    # confusion = confusion_matrix(test_true_labels, test_preds_labels)

    # print('F1_score : {: .10f}'.format(f1))
    # print('Recall : {: .10f}'.format(recall))
    # print('Precision : {: .10f}'.format(precision))
    # print('AUC : {: .10f}'.format(auc))
    # print(confusion)

    return test_acc, f1


def dual_model_test(path, prediction_dataloader, device, fold):
    ########################## TEST #################################
    test_preds_labels_list = np.array([])
    test_true_labels_list = np.array([])
    model = torch.load(path + '/multimodal' + str(fold) + '.pt',
                       map_location=device)
    model.eval()
    model.to(device)
    for test_batch in prediction_dataloader:
        input_ids_url = test_batch[0].to(device)
        attention_mask_url = test_batch[1].to(device)
        labels_url = test_batch[2].to(device)
        input_ids_content = test_batch[3].to(device)
        attention_mask_content = test_batch[4].to(device)
        labels_content = test_batch[5].to(device)
        with torch.no_grad():
            outputs = model(input_ids_url, attention_mask_url, labels_url, input_ids_content,
                            attention_mask_content, labels_content)
        pred_labels = (outputs >= 0.5).float()
        labels = labels_url.unsqueeze(1).float()
        test_preds_labels_list = np.append(test_preds_labels_list, pred_labels.to('cpu').numpy())
        test_true_labels_list = np.append(test_true_labels_list, labels.to('cpu').numpy())
    test_preds_labels = np.array(test_preds_labels_list)
    test_true_labels = np.array(test_true_labels_list)
    compare = cal_acc(test_preds_labels, test_true_labels)
    test_acc = sum(compare) / len(compare)
    # print(f'Test dataset Accuracy : {test_acc:.4f}')

    f1 = f1_score(test_true_labels, test_preds_labels, average='weighted')
    # recall = recall_score(test_true_labels, test_preds_labels, average='weighted')
    # precision = precision_score(test_true_labels, test_preds_labels, average='weighted')
    # confusion = confusion_matrix(test_true_labels, test_preds_labels)
    #
    # print('F1_score : {: .10f}'.format(f1))
    # print('Recall : {: .10f}'.format(recall))
    # print('Precision : {: .10f}'.format(precision))
    # print(confusion)

    return test_acc, f1