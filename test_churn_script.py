""" Unit test of churn_library.py module with pytest """

import logging
from churn_library import *
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


############################ Fixtures ##############################
@pytest.fixture
def path():
    """
    Fixture - The test function test_import() will
    use the return of path() as an argument
    """
    return r"./data/bank_data.csv"

@pytest.fixture
def dataset(path):
    """
    Fixture - The test functions will
    use the return of dataset as an argument
    """
    return import_data(path)

############################ Unit Tests ##############################
def test_import(path):
    '''
test data import - this example is completed for you to assist with the other test functions
    '''

    try:
        data = import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(dataset):
    '''
    test perform eda function
    '''
    try:
        perform_eda(dataset)
        logging.info("Testing perform_eda - SUCCESS")

    except Exception as err:
        logging.error("Testing perform_eda - ERROR")
        raise err


def test_encoder_helper(dataset):
    '''
    test encoder helper
    '''
    cat = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
    try:

        encoder_helper(dataset, cat)
        logging.info("Testing encoder_helper - SUCCESS")

    except Exception as err:
        logging.error("Testing encoder_helper - ERROR")
        raise err


def test_perform_feature_engineering(dataset):
    '''
    test perform_feature_engineering
    '''
    try:
        perform_feature_engineering(dataset)
        logging.info("Testing perform_feature_engineering - SUCCESS")


    except Exception as err:
        raise err


def test_train_models(dataset):
    '''
    test train_models
    '''
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataset)
    try:

        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing test_train_models - SUCCESS")

    except Exception as err:
        logging.error("Testing test_train_models - ERROR")
        raise err


if __name__ == "__main__":
    source = r"./data/bank_data.csv"
    test_import(source)
    test_eda(import_data(source))
    test_encoder_helper(import_data(source))
    test_perform_feature_engineering(import_data(source))
    test_train_models(import_data(source))

