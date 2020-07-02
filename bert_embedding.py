#!/home/junekyu/anaconda3 python
# -*- coding: utf-8 -*-

import numpy as np
import time
import datetime
import random

import torch
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

import pdb


def tokenize(tokenizer, sentences, next_sentences, MAX_LEN):

    input_ids = []
    attention_masks = []

    print("encoding...")
    for i, (sent, next_sent) in enumerate(zip(sentences, next_sentences)):
        if i % 1000 == 0: print("encoding " + str(i) + "th sentence")
        _encoded = tokenizer.encode_plus(
            sent,
            text_pair=next_sent,
            add_special_tokens=True,
            max_length=MAX_LEN,
            #  pad_to_max_length=True,
            return_tensors='pt',
        )
        #  pdb.set_trace()
        #  input_ids = _encoded['input_ids'][0].numpy()
        encoded_sent = _encoded['input_ids'][0]
        #  attention_mask = _encoded['special_tokens_mask']

        #  encoded_sent -> ['special_tokens_mask'], ['input_ids'], ['token_type_ids']
        input_ids.append(encoded_sent)
        #  attention_masks.append(attention_mask)

    input_ids = pad_sequences(input_ids,
                              maxlen=MAX_LEN,
                              dtype="long",
                              value=0,
                              truncating="post",
                              padding="post")
    #  attention_mask = _encoded['special_tokens_mask']

    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    #  pdb.set_trace()

    return input_ids, attention_masks


def data_processing(data_x, data_title, index2word):

    x = []

    for i in range(len(data_x)):  # for all data_x
        if i % 1000 == 0:
            print("processed " + str(i) + "th sentence")
        sentence = ""
        for j in range(len(data_x[i])):  # for each input
            word = index2word[data_x[i][j]]
            sentence += word + " "
        x.append(sentence)

    title = []
    for i in range(len(data_title)):
        if i % 1000 == 0:
            print("processed " + str(i) + "th sentence")
        sentence = ""
        for j in range(len(data_title[i])):
            word = index2word[data_title[i][j]]
            sentence += word + " "
        title.append(sentence)

    return x, title


def flat_accuracy(pred, labels):
    pred_flat = np.argmax(pred, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def bert_tune_and_test(train_data, val_data, test_data, num_labels, num_epochs,
                       batch_size):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data,
                                sampler=val_sampler,
                                batch_size=batch_size)

    embedding_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False)

    embedding_model.cuda()

    optimizer = AdamW(embedding_model.parameters(), lr=2e-5, eps=1e-8)

    epochs = num_epochs

    total_steps = len(train_dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []

    # ===============
    # fine tuning
    # ===============

    for epoch_i in range(0, epochs):
        t0 = time.time()
        total_loss = 0
        temp_loss = 0
        embedding_model.train()

        #  pdb.set_trace()

        # ===============
        # training
        # ===============

        print("training...")
        print("")

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_attn_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            embedding_model.zero_grad()

            outputs = embedding_model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_attn_mask,
                                      labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            temp_loss += loss.item()
            if step % 40 == 0 and not step == 0:
                print('  Loss {:>5,}'.format(temp_loss / 40))
                temp_loss = 0

            torch.nn.utils.clip_grad_norm_(
                embedding_model.parameters(),
                1.0)  # prevent the "exploding gradient"

            optimizer.step()

            # update the learning rate (in transformer architechture)
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() -
                                                              t0)))
        # ===============
        # validation
        # ===============

        print("validation...")
        print("")

        t0 = time.time()

        embedding_model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_step, nb_eval_examples = 0, 0

        for batch in val_dataloader:

            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_attn_mask, b_labels = batch

            # assigning to device
            with torch.no_grad():
                outputs = embedding_model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_attn_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy

            nb_eval_step += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_step))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    # ===============
    # testing
    # ===============
    #  pdb.set_trace()

    print("testing...")
    print("")

    test_sampler = SequntialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 sampler=test_sampler,
                                 batch_size=batch_size)

    embedding_model.eval()

    predictions, true_labels = [], []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = embedding_model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask)

        logits = output[0]
        logits = logits.detacth().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    accuracy_set = []

    for i in range(len(true_labels)):

        acc = flat_accuracy(predictions[i], true_labels[i])
        accuracy_set.append(acc)

    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    test_accuracy = flat_accuracy(flat_predictions, flat_true_labels)
    print("  Accuracy: {0:.2f}".format(test_accuracy))


def bert(dataset):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)

    #  voca = dataset['voca']
    train_x = dataset['train_x']
    train_title = dataset['train_title']
    train_y = dataset['train_y']

    dev_x = dataset['dev_x']
    dev_title = dataset['dev_title']
    dev_y = dataset['dev_y']

    test_x = dataset['test_x']
    test_title = dataset['test_title']
    test_y = dataset['test_y']

    #  pad_index = dataset['pad_index']
    index2word = dataset['index2word']

    train_x, train_title = data_processing(train_x, train_title, index2word)
    dev_x, dev_title = data_processing(dev_x, dev_title, index2word)
    test_x, test_title = data_processing(test_x, test_title, index2word)

    MAX_LEN = 512

    train_inputs, train_attn_masks = tokenize(tokenizer, train_x, train_title,
                                              MAX_LEN)
    dev_inputs, dev_attn_masks = tokenize(tokenizer, dev_x, dev_title, MAX_LEN)
    test_inputs, test_attn_masks = tokenize(tokenizer, test_x, test_title,
                                            MAX_LEN)

    #  pdb.set_trace()

    train_inputs = torch.tensor(train_inputs)
    train_attn_masks = torch.tensor(train_attn_masks)
    train_labels = torch.tensor(train_y)
    dev_inputs = torch.tensor(dev_inputs)
    dev_attn_masks = torch.tensor(dev_attn_masks)
    dev_labels = torch.tensor(dev_y)
    test_inputs = torch.tensor(test_inputs)
    test_attn_masks = torch.tensor(test_attn_masks)
    test_labels = torch.tensor(test_y)

    train_data = TensorDataset(train_inputs, train_attn_masks, train_labels)
    val_data = TensorDataset(dev_inputs, dev_attn_masks, dev_labels)
    test_data = TensorDataset(test_inputs, test_attn_masks, test_labels)

    #  pdb.set_trace()

    bert_tune_and_test(train_data,
                       val_data,
                       test_data,
                       num_labels=2,
                       num_epochs=5,
                       batch_size=8)
