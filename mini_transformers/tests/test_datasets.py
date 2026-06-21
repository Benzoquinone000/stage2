from mini_transformers.data.preprocessing import train_valid_split
from mini_transformers.data.collators import (
    DataCollatorForMaskedLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
)
from mini_transformers.data import BertPreTrainingDataset
from mini_transformers.tokenization import BasicTokenizer, Vocab


def test_train_valid_split_size():
    train, valid = train_valid_split(list(range(10)), valid_ratio=0.2, seed=1)
    assert len(train) == 8
    assert len(valid) == 2


def test_seq2seq_collator_shifts_decoder_inputs():
    collator = DataCollatorForSeq2Seq(pad_token_id=0, start_token_id=2)
    batch = collator([{"input_ids": [2, 5, 3], "labels": [2, 7, 3]}])
    assert batch["decoder_input_ids"].tolist() == [[2, 2, 7]]
    assert batch["labels"].tolist() == [[2, 7, 3]]


def test_permutation_lm_collator_adds_perm_mask():
    collator = DataCollatorForPermutationLanguageModeling()
    batch = collator([{"input_ids": [4, 5, 6], "labels": [4, 5, 6]}])
    assert batch["perm_mask"].shape == (1, 3, 3)
    assert batch["labels"].tolist() == [[4, 5, 6]]


def test_masked_lm_collator_masks_at_least_one_token():
    collator = DataCollatorForMaskedLanguageModeling(mask_token_id=4, mlm_probability=0.0)
    batch = collator([{"input_ids": [2, 5, 6, 3], "attention_mask": [1, 1, 1, 1]}])
    assert (batch["labels"] != -100).any()
    assert 4 in batch["input_ids"].tolist()[0]


def test_bert_pretraining_dataset_pair_fields():
    vocab = Vocab(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "again"])
    tokenizer = BasicTokenizer(vocab)
    dataset = BertPreTrainingDataset(["hello world", "hello again", "world again"], tokenizer, max_length=8)
    item = dataset[0]
    assert "token_type_ids" in item
    assert "next_sentence_labels" in item
