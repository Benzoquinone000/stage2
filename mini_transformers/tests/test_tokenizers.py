from mini_transformers.tokenization import (
    BPETokenizer,
    BasicTokenizer,
    Vocab,
    WordPieceTokenizer,
    load_tokenizer,
    save_tokenizer,
)


def test_basic_tokenizer_ids():
    vocab = Vocab(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", ",", "world"])
    tokenizer = BasicTokenizer(vocab)
    encoded = tokenizer.encode("Hello, world", add_special_tokens=False)
    assert encoded.input_ids == [4, 5, 6]


def test_vocab_from_tokens_keeps_special_tokens_first():
    vocab = Vocab.from_tokens(["b", "a", "b"])
    assert vocab.tokens[:2] == ["[PAD]", "[UNK]"]
    assert "a" in vocab
    assert "b" in vocab


def test_bpe_tokenizer_train_save_load(tmp_path):
    tokenizer = BPETokenizer.train(["low lower lowest", "newer lower"], vocab_size=30, min_pair_freq=1)
    pieces = tokenizer.tokenize("lower")
    assert pieces
    assert tokenizer.decode(tokenizer.convert_tokens_to_ids(pieces)) == "lower"
    tokenizer.vocab.save(tmp_path / "vocab.txt")
    tokenizer.save_merges(tmp_path / "merges.txt")
    loaded = BPETokenizer.from_files(tmp_path / "vocab.txt", tmp_path / "merges.txt")
    assert loaded.tokenize("lower") == pieces


def test_bpe_tokenizer_round_trip_multi_word(tmp_path):
    tokenizer = BPETokenizer.train(["low lower lowest", "newer lower"], vocab_size=40, min_pair_freq=1)
    save_tokenizer(tokenizer, tmp_path)
    loaded = load_tokenizer(tmp_path)
    ids = loaded.convert_tokens_to_ids(loaded.tokenize("low lower"))
    assert loaded.decode(ids) == "low lower"


def test_wordpiece_train_and_decode():
    tokenizer = WordPieceTokenizer.train(["hello world"], vocab_size=20)
    ids = tokenizer.convert_tokens_to_ids(["hello", "##s"])
    decoded = tokenizer.decode(ids)
    assert "hello" in decoded


def test_wordpiece_tokenizer_save_load_type(tmp_path):
    tokenizer = WordPieceTokenizer.train(["playing player"], vocab_size=30)
    save_tokenizer(tokenizer, tmp_path)
    loaded = load_tokenizer(tmp_path)
    assert isinstance(loaded, WordPieceTokenizer)
    assert loaded.tokenize("playing") == tokenizer.tokenize("playing")
