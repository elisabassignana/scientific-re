import torch
from sklearn.metrics import f1_score
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

class DatasetMaper(Dataset):

    def __init__(self, s, p1, p2, y):
        self.s = s
        self.p1 = p1
        self.p2 = p2
        self.y = y

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.s[idx], self.p1[idx], self.p2[idx], self.y[idx]

class Run:

    def __init__(self, model, prepared_data, params):

        self.data = prepared_data

        # Prepare batches
        train = DatasetMaper(self.data.train.sentence, self.data.train.pos1, self.data.train.pos2, self.data.train.y)
        dev = DatasetMaper(self.data.dev.sentence, self.data.dev.pos1, self.data.dev.pos2, self.data.dev.y)
        test = DatasetMaper(self.data.test.sentence, self.data.test.pos1, self.data.test.pos2, self.data.test.y)
        self.loader_train = DataLoader(train, batch_size=params.batch_size)
        self.loader_dev = DataLoader(dev, batch_size=params.batch_size)
        self.loader_test = DataLoader(test, batch_size=params.batch_size)

        self.device = params.device
        self.len_sentences = params.len_sentences
        self.relations = params.relations
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=params.learning_rate)
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(self.compute_loss_weight()).to(self.device))
        self.tokenizerBert = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.modelBert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.modelBert.to(self.device)


    def train(self):
        train_losses, dev_losses, macro_fscores_train, macro_fscores_dev = [], [], [], []
        self.model.to(self.device)

        for epoch in range(self.epochs):
            running_loss = 0

            # f-score train
            tot_predictions_train, tot_targets_train = [], []

            for s_batch, p1_batch, p2_batch, y_batch in self.loader_train:

                # Train f-score
                tot_targets_train = tot_targets_train + y_batch.tolist()

                # BERT
                sentences_list = [[token for token in sentence.split()] for sentence in s_batch]
                tokens = self.tokenizerBert(sentences_list, return_offsets_mapping=True, is_split_into_words=True, padding='max_length', truncation=True, max_length=self.len_sentences)
                encoded_sentences = []
                for id_list, offset_list in zip(tokens['input_ids'], tokens['offset_mapping']):
                    encoded_sentence = []
                    for id, offset in zip(id_list, offset_list):
                        if offset[0] == 0 and offset[1] != 0:
                            encoded_sentence.append(id)
                    encoded_sentence.extend([0] * (self.len_sentences - len(encoded_sentence)))
                    encoded_sentences.insert(len(encoded_sentences), encoded_sentence)
                embedsbert = self.modelBert(torch.LongTensor(encoded_sentences).to(self.device))[0]

                # Move input tensors to the device
                p1_batch, p2_batch, y_batch = p1_batch.to(self.device), p2_batch.to(self.device), torch.LongTensor(y_batch).to(self.device)

                # Model prediction
                self.optimizer.zero_grad()
                prediction = self.model(embedsbert, p1_batch, p2_batch)

                # f-score train
                ps = torch.exp(prediction)
                top_p, top_class = ps.topk(1, dim=1)
                for elem in top_class:
                    tot_predictions_train.append(int(elem))

                # Loss and backward step
                loss = self.loss(prediction, y_batch)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            else:
                # Update train f-score
                macro_fscore_train = round(f1_score(tot_targets_train, tot_predictions_train, average="macro") * 100, 2)
                macro_fscores_train.append(macro_fscore_train)

                # dev

                dev_loss = 0

                # dev f-score
                tot_predictions_dev, tot_targets_dev = [], []

                with torch.no_grad():
                    self.model.eval()

                    for s_batch_dev, p1_batch_dev, p2_batch_dev, y_batch_dev in self.loader_dev:

                        tot_targets_dev = tot_targets_dev + y_batch_dev.tolist()

                        # BERT
                        sentences_list_dev = [[token for token in sentence.split()] for sentence in s_batch_dev]
                        tokens_dev = self.tokenizerBert(sentences_list_dev, return_offsets_mapping=True, is_split_into_words=True, padding='max_length', truncation=True, max_length=self.len_sentences)
                        encoded_sentences_dev = []
                        for id_list, offset_list in zip(tokens_dev['input_ids'], tokens_dev['offset_mapping']):
                            encoded_sentence_dev = []
                            for id, offset in zip(id_list, offset_list):
                                if offset[0] == 0 and offset[1] != 0:
                                    encoded_sentence_dev.append(id)
                            encoded_sentence_dev.extend([0] * (self.len_sentences - len(encoded_sentence_dev)))
                            encoded_sentences_dev.insert(len(encoded_sentences_dev), encoded_sentence_dev)
                        embedsbert_dev = self.modelBert(torch.LongTensor(encoded_sentences_dev).to(self.device))[0]

                        # Move input tensors to the device
                        p1_batch_dev, p2_batch_dev,y_batch_dev = p1_batch_dev.to(self.device), p2_batch_dev.to(self.device), torch.LongTensor(y_batch_dev).to(self.device)

                        # Model prediction
                        prediction = self.model(embedsbert_dev, p1_batch_dev, p2_batch_dev)

                        dev_loss += self.loss(prediction, y_batch_dev)

                        # From the model prediction to the original class
                        ps = torch.exp(prediction)
                        top_p, top_class = ps.topk(1, dim=1)
                        for elem in top_class:
                            tot_predictions_dev.append(int(elem))

                self.model.train()

                # Update f-score dev
                macro_fscore_dev = round(f1_score(tot_targets_dev, tot_predictions_dev, average="macro") * 100, 2)
                macro_fscores_dev.append(macro_fscore_dev)

                # Update train and dev loss
                train_losses.append(running_loss)
                dev_losses.append(dev_loss)

                print("Epoch: {}/{}".format(epoch + 1, self.epochs),
                      "Training Loss: {:.3f}".format(train_losses[-1]),
                      "Dev Loss: {:.3f}".format(dev_losses[-1]),
                      "Macro f-score train: {}".format(macro_fscore_train),
                      "Macro f-score dev: {}".format(macro_fscore_dev))

        return macro_fscores_train, macro_fscores_dev


    def test(self):

        tot_predictions_test, tot_targets_test = [], []

        with torch.no_grad():
            self.model.eval()

            for s_batch_test, p1_batch_test, p2_batch_test, y_batch_test in self.loader_test:

                tot_targets_test = tot_targets_test + y_batch_test.tolist()

                # BERT
                sentences_list_test = [[token for token in sentence.split()] for sentence in s_batch_test]
                tokens_test = self.tokenizerBert(sentences_list_test, return_offsets_mapping=True, is_split_into_words=True, padding='max_length', truncation=True, max_length=self.len_sentences)
                encoded_sentences_test = []
                for id_list, offset_list in zip(tokens_test['input_ids'], tokens_test['offset_mapping']):
                    encoded_sentence_test = []
                    for id, offset in zip(id_list, offset_list):
                        if offset[0] == 0 and offset[1] != 0:
                            encoded_sentence_test.append(id)
                    encoded_sentence_test.extend([0] * (self.len_sentences - len(encoded_sentence_test)))
                    encoded_sentences_test.insert(len(encoded_sentences_test), encoded_sentence_test)
                embedsbert_test = self.modelBert(torch.LongTensor(encoded_sentences_test).to(self.device))[0]

                p1_batch_test, p2_batch_test, y_batch_test = p1_batch_test.to(self.device), p2_batch_test.to(self.device), torch.LongTensor(y_batch_test).to(self.device)

                prediction = self.model(embedsbert_test, p1_batch_test, p2_batch_test)
                ps = torch.exp(prediction)
                top_p, top_class = ps.topk(1, dim=1)
                for elem in top_class:
                    tot_predictions_test.append(int(elem))

        return round(f1_score(tot_targets_test, tot_predictions_test, average="macro") * 100, 2)


    def compute_loss_weight(self):

        relation_count = [0] * len(self.relations)

        for label in self.data.train.y:
            relation_count[label] += 1

        tot = sum(relation_count)
        relation_count = [1 / (elem / tot) if elem > 0 else 1 for elem in relation_count]

        return relation_count