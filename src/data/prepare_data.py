import numpy

class Preparedata:

    def __init__(self, params):

        self.dataset_train = params.train
        self.relations_train = params.train_relations
        self.dataset_dev = params.dev
        self.relations_dev = params.dev_relations
        self.dataset_test = params.test
        self.relations_test = params.test_relations

        self.train = Data()
        self.dev = Data()
        self.test = Data()

    def prepare_data(self, param):
        self.lookuptables = LookUpTables(param)

        sentences, pos1, pos2, y = self.generate_sentences(param.len_sentences, self.dataset_train, self.relations_train)
        self.train.set_data(sentences, pos1, pos2, y)

        sentences, pos1, pos2, y = self.generate_sentences(param.len_sentences, self.dataset_dev, self.relations_dev)
        self.dev.set_data(sentences, pos1, pos2, y)

        sentences, pos1, pos2, y = self.generate_sentences(param.len_sentences, self.dataset_test, self.relations_test)
        self.test.set_data(sentences, pos1, pos2, y)

    # Generate the instances
    def generate_sentences(self, len_sentences, dataset_file, relation_file):

        relations = []
        with open(relation_file) as r:
            id_sentence = r.readline().split('\t')[0].split('.')[0]
            sentence = []
            r.seek(0)
            for line in r:
                relation_parts = line.strip().split('\t')
                if relation_parts[0].split('.')[0] == id_sentence:
                    sentence.append((relation_parts[0], relation_parts[1], relation_parts[2]))
                else:
                    relations.append(sentence)
                    sentence = [(relation_parts[0], relation_parts[1], relation_parts[2])]
                    id_sentence = relation_parts[0].split('.')[0]
            relations.append(sentence)

        first = True
        sentencetot, pos1tot, pos2tot, ytot = [], [], [], []

        for rel_abstract in relations:
            with open(dataset_file) as d:
                tokens_abstract = [
                    (t.strip().split('\t')[0], int(t.strip().split('\t')[1]), int(t.strip().split('\t')[2]),
                     t.strip().split('\t')[3], t.strip().split('\t')[4]) for t in d if
                    t.strip().split('\t')[0] == rel_abstract[0][0].split('.')[0]]
            for rel in rel_abstract:
                ent1 = [elem for elem in tokens_abstract if elem[4] == rel[0]][0]
                ent2 = [elem for elem in tokens_abstract if elem[4] == rel[1]][0]
                tokens_sentence = []
                if ent1[1] == ent2[1]:
                    tokens_sentence = [elem for elem in tokens_abstract if elem[1] == ent1[1]]

                if len(tokens_sentence) <= len_sentences:
                    sentence = ''
                    pos1_emb, pos2_emb = [], []

                    for token in tokens_sentence:

                        # sentence
                        sentence += token[3] + ' '

                        # distance
                        pos1_token = token[2] - ent1[2]
                        if pos1_token < -10 or pos1_token > 10:
                            pos1_emb.append(self.lookuptables.position2id['LONG_DISTANCE'])
                        else:
                            pos1_emb.append(self.lookuptables.position2id[pos1_token])
                        pos2_token = token[2] - ent2[2]
                        if pos2_token < -10 or pos2_token > 10:
                            pos2_emb.append(self.lookuptables.position2id['LONG_DISTANCE'])
                        else:
                            pos2_emb.append(self.lookuptables.position2id[pos2_token])

                    pos1_emb.extend([self.lookuptables.position2id['<PAD>']] * (len_sentences - len(pos1_emb)))
                    pos2_emb.extend([self.lookuptables.position2id['<PAD>']] * (len_sentences - len(pos2_emb)))

                    if first == True:
                        sentencetot = [sentence]
                        pos1tot = [pos1_emb]
                        pos2tot = [pos2_emb]
                        first = False
                    else:
                        sentencetot = numpy.concatenate((sentencetot, [sentence]), axis=0)
                        pos1tot = numpy.concatenate((pos1tot, [pos1_emb]), axis=0)
                        pos2tot = numpy.concatenate((pos2tot, [pos2_emb]), axis=0)
                    ytot.append(self.lookuptables.relation2id[rel[2]])

        return sentencetot, pos1tot, pos2tot, ytot


class Data:

    def __init__(self):
        self.sentence = None
        self.pos1 = None
        self.pos2 = None
        self.y = None

    def set_data(self, sentence, pos1, pos2, y):
        self.sentence = sentence
        self.pos1 = pos1
        self.pos2 = pos2
        self.y = y


class LookUpTables:

    def __init__(self, params):

        # positions
        self.position2id = {position: idtable for position, idtable in zip([*range(params.min_position, params.max_position + 2, 1)], [*range(2, params.num_positions, 1)])}
        self.position2id['LONG_DISTANCE'] = 1
        self.position2id['<PAD>'] = 0
        self.id2position = {v: k for k, v in self.position2id.items()}

        # relations
        self.relation2id = {item: index for index, item in enumerate(params.relations, 0)}
        self.id2relation = {v: k for k, v in self.relation2id.items()}