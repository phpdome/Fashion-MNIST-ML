from src.data_loader import load_data

def test_load_data_shapes():
    train, test = load_data()
    assert len(train) > 0
    assert len(test) > 0

def test_sample_shape():
    train, _ = load_data()
    x, y = train[0]
    assert tuple(x.shape) == (1, 28, 28)
    assert isinstance(y, int)
