import pandas as pd
from src.utils.explainer import TechnicalExplainer

print(f"TechnicalExplainer type: {type(TechnicalExplainer)}")
print(f"Attributes: {dir(TechnicalExplainer)}")

try:
    df = pd.DataFrame({'close': [1, 2, 3]})
    TechnicalExplainer.explain(df)
    print("Execution successful")
except Exception as e:
    print(f"Execution failed: {e}")
