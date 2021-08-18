import random


class Data:
    def __init__(self, data_dir="data/FB15k/"):
        ###############################################entity type#######################################################
        self.train_type_data = self.load_type_data(data_dir, "Entity_Type_train")
        self.valid_type_data = self.load_type_data(data_dir, "Entity_Type_valid")
        self.test_type_data = self.load_type_data(data_dir, "Entity_Type_test")

        self.type_data = self.train_type_data + self.valid_type_data + self.test_type_data

        self.types = self.get_types(self.type_data)
        self.entities = self.get_entities(self.type_data)

        self.entity_idxs = self.encode_entity_to_id(data_dir)
        self.idxs_entity = {v: k for k, v in self.entity_idxs.items()}

        self.entity_type_idxs = self.encode_type_to_id(data_dir)
        self.idxs_entity_type = {v: k for k, v in self.entity_type_idxs.items()}

        self.train_idxs = self.get_type_data_idxs(self.train_type_data)
        random.shuffle(self.train_idxs)
        self.valid_idxs = self.get_type_data_idxs(self.valid_type_data)
        self.test_idxs = self.get_type_data_idxs(self.test_type_data)

        self.over_data = self.train_idxs + self.valid_idxs + self.test_idxs

        self.type_to_entity_dict = self.get_type_to_entity(self.train_idxs)
        self.type_to_entity_dict2 = self.get_type_to_entity(self.valid_idxs)
        self.type_to_entity_dict3 = self.get_type_to_entity(self.test_idxs)

        self.entity_to_type_dict = self.get_type_data_idxs_dict(self.train_idxs)
        self.entity_to_type_dict2 = self.get_type_data_idxs_dict(self.valid_idxs)
        self.entity_to_type_dict3 = self.get_type_data_idxs_dict(self.test_idxs)

        self.test_data_1_1, self.test_data_1_N = self.get_1_1_OR_1_N_test_data(self.entity_to_type_dict3)
        random.shuffle(self.test_data_1_1)
        random.shuffle(self.test_data_1_N)
        pass

    def encode_entity_to_id(self, data_dir):
        entity2id = {}
        total_entity_num = 0
        with open(data_dir + 'entity2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ent, ent2id = line.strip().split("\t")
                entity2id[ent] = int(ent2id)
        return entity2id

    def encode_type_to_id(self, data_dir):
        type2id = {}
        with open(data_dir + 'type2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                t, t2id = line.strip().split("\t")
                type2id[t] = int(t2id)
        return type2id

    def load_type_data(self, data_dir, data_type):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding='utf-8') as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    def get_types(self, data):
        types = sorted(list(set([d[1] for d in data])))
        return types

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data])))
        return entities

    def get_type_data_idxs(self, data):
        entity_type_data_idxs = [(self.entity_idxs[data[i][0]], self.entity_type_idxs[data[i][1]]) for i in
                                 range(len(data))]
        return entity_type_data_idxs

    def get_type_data_idxs_dict(self, data):
        entity_type = {}
        for temp in data:
            entity_type.setdefault(temp[0], set()).add(temp[1])
        return entity_type

    def get_type_to_entity(self, data):
        type2entity = {}
        for temp in data:
            type2entity.setdefault(temp[1], set()).add(temp[0])
        return type2entity

    def get_1_1_OR_1_N_test_data(self, data):
        test_data_1_1 = []
        test_data_1_N = []
        for k in data.keys():
            if (len(data[k]) == 1):
                temp = (k, list(data[k])[0])
                test_data_1_1.append(temp)
            else:
                for t in data[k]:
                    temp = (k, t)
                    test_data_1_N.append(temp)
        return test_data_1_1, test_data_1_N
