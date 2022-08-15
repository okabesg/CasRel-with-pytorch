from torch import nn


def get_feature(token_feature, idx):
    pass


class subject_model(nn.Module):
    def __init__(self, bert_model, input_dim):
        super().__init__()
        self.bert_model = bert_model
        self.lr_subhead = nn.Linear(input_dim, 1)
        self.sigmoid_subhead = nn.Sigmoid()
        self.lr_subtail = nn.Linear(input_dim, 1)
        self.sigmoid_subtail = nn.Sigmoid()

    def forward(self, token_ids, seg_ids):
        token_feature = self.bert_model(token_ids, seg_ids)
        pred_sub_heads = self.sigmoid_subhead(self.lr_subhead(token_feature))
        pred_sub_tails = self.sigmoid_subtail(self.lr_subtail(token_feature))
        return pred_sub_heads, pred_sub_tails

class object_model(nn.Module):
    def __init__(self, bert_model, input_dim, num_rels):
        super().__init__()
        self.bert_model = bert_model
        self.lr_objhead = nn.Linear(input_dim, num_rels)
        self.sigmoid_objhead = nn.Sigmoid()
        self.lr_objtail = nn.Linear(input_dim, num_rels)
        self.sigmoid_objtail = nn.Sigmoid()

    def forward(self, token_ids, seg_ids, sub_head_in, sub_tail_in):
        token_feature = self.bert_model(token_ids, seg_ids)
        head_feature = get_feature(token_feature, sub_head_in)
        tail_feature = get_feature(token_feature, sub_tail_in)
        sub_feature = (head_feature + tail_feature) / 2
        token_feature = token_feature + sub_feature

        pred_obj_heads = self.sigmoid_objhead(self.lr_objhead(token_feature))
        pred_obj_tails = self.sigmoid_objtail(self.lr_objtail(token_feature))
        return pred_obj_heads, pred_obj_tails

