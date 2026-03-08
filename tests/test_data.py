import pytest
import pandas as pd
import os

@pytest.fixture
def iris_df():
    return pd.read_csv('data/iris.csv')

def test_file_exists():
    assert os.path.exists('data/iris.csv'), "Data file missing"

def test_schema(iris_df):
    expected_cols = ['sepal length (cm)', 'sepal width (cm)',
                     'petal length (cm)', 'petal width (cm)', 'target']
    for col in expected_cols:
        assert col in iris_df.columns, f"Missing column: {col}"

def test_no_missing_values(iris_df):
    assert iris_df.isnull().sum().sum() == 0, "Data has missing values"

def test_row_count(iris_df):
    assert len(iris_df) == 150, "Expected 150 rows"

def test_target_classes(iris_df):
    assert set(iris_df['target'].unique()) == {0, 1, 2}, "Wrong target classes"

def test_feature_ranges(iris_df):
    assert iris_df['sepal length (cm)'].between(4, 8).all()
    assert iris_df['petal length (cm)'].between(1, 7).all()
