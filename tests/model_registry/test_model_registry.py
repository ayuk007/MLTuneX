import unittest
from mltunex.model_registry.model_registry import Model_Registry

class TestModelRegistry(unittest.TestCase):
    def test_get_model_registry(self):
        # Test for sklearn model registry
        registry = Model_Registry.get_model_registry("sklearn")
        self.assertIsNotNone(registry)
        # self.assertEqual(registry.__name__, "SkLearn_Model_Registry")

        # Test for unsupported model library
        with self.assertRaises(ValueError):
            Model_Registry.get_model_registry("unsupported_library")

    def test_get_classification_models(self):
        # Test for sklearn classification models
        registry = Model_Registry.get_model_registry("sklearn")
        models = registry.get_classification_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    def test_get_regression_models(self):
        # Test for sklearn regression models
        registry = Model_Registry.get_model_registry("sklearn")
        models = registry.get_regression_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

if __name__ == "__main__":
    unittest.main()