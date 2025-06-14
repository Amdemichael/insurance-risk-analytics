def test_load_data():
    from src.eda import load_data
    df = load_data('data/MachineLearningRating_v3.txt')
    assert not df.empty