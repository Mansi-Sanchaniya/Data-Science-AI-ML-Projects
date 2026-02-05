

import os
import dspy
import pandas as pd
from dotenv import load_dotenv
# Load .env file
load_dotenv()


#keyword dictionaries(domain knowledge)
RISK_KW = ["risk", "volatility", "uncertainty", "inflation", "forex", "headwind"]
REG_KW = ["sebi", "rbi", "regulation", "compliance", "audit"]
GUIDANCE_KW = ["expect", "guidance", "outlook", "forecast", "looking ahead"]
LIABILITY_KW = ["litigation", "lawsuit", "tax dispute", "penalty"]
COST_KW = ["cost", "expense", "margin pressure", "input cost"]
REVENUE_UP = ["revenue growth", "strong demand", "growth"]
REVENUE_DOWN = ["revenue decline", "weak demand", "slowdown"]
CAPITAL_KW = ["capex", "buyback", "dividend", "debt"]


#weak label generator
def extract_signals(text):
    t = text.lower()

    def find(kws):
        found = [k for k in kws if k in t]
        return "; ".join(found) if found else ""

    revenue = "Flat"
    if find(REVENUE_UP):
        revenue = "Growing"
    elif find(REVENUE_DOWN):
        revenue = "Declining"

    sentiment = "Neutral"
    if "strong" in t or "confident" in t:
        sentiment = "Positive"
    elif "challenging" in t or "pressure" in t:
        sentiment = "Cautious"

    return {
        "material_risks": find(RISK_KW),
        "regulatory_obligations": find(REG_KW),
        "forward_guidance": find(GUIDANCE_KW),
        "contingent_liabilities": find(LIABILITY_KW),
        "cost_pressures": find(COST_KW),
        "revenue_trend": revenue,
        "capital_allocation": find(CAPITAL_KW),
        "management_sentiment": sentiment
    }


rows = []

for item in dataset[:50]:
    signals = extract_signals(item["text"])
    row = {"text": item["text"][:3000]}  # truncate safely
    row.update(signals)
    rows.append(row)

df = pd.DataFrame(rows)
df.head()


df.to_csv("earnings_ground_truth.csv", index=False)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Method 1: Direct LiteLLM format (most reliable)
lm = dspy.LM(
    model='gemini/gemini-2.5-flash-lite',  # Or gemini-1.5-flash-8b
    api_key=GOOGLE_API_KEY,
    api_base="https://generativelanguage.googleapis.com/v1beta",
    max_tokens=800,
    temperature=0
)

dspy.settings.configure(lm=lm)

class FinancialInsightExtraction(dspy.Signature):

    document = dspy.InputField(desc="Financial disclosure text")

    material_risks = dspy.OutputField(desc="Key financial risks")
    regulatory_obligations = dspy.OutputField(desc="Mandatory regulatory obligations")
    forward_guidance = dspy.OutputField(desc="Statements about future performance")
    contingent_liabilities = dspy.OutputField(desc="Potential financial liabilities")


class FinancialExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(FinancialInsightExtraction)

    def forward(self, document):
        return self.predict(document=document)

extractor = FinancialExtractor()

sample_text = """
We continue to see volatility in foreign exchange markets, particularly in our European operations.
Despite macroeconomic headwinds, management expects revenue growth in the mid-single digits in FY25.
We are currently contesting certain tax assessments, which remain under litigation.
"""

result = extractor(document=sample_text)

print(result)


import pandas as pd

df = pd.read_csv("earnings_ground_truth.csv")

trainset = []

for _, row in df.iterrows():
    trainset.append(
        dspy.Example(
            document=row["text"],
            material_risks=row["material_risks"],
            regulatory_obligations=row["regulatory_obligations"],
            forward_guidance=row["forward_guidance"],
            contingent_liabilities=row["contingent_liabilities"],
        ).with_inputs("document")
    )


def finance_metric(example, prediction, trace=None):
    score = 0

    if prediction.material_risks and example.material_risks:
        score += 1

    if prediction.regulatory_obligations and example.regulatory_obligations:
        score += 1

    if prediction.forward_guidance and example.forward_guidance:
        score += 1

    if prediction.contingent_liabilities and example.contingent_liabilities:
        score += 1

    return score / 4


from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(
    metric=finance_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=8
)

optimized_extractor = teleprompter.compile(
    FinancialExtractor(),
    trainset=trainset
)

baseline = extractor(document=sample_text)
optimized = optimized_extractor(document=sample_text)

print("BASELINE:\n", baseline)
print("\nOPTIMIZED:\n", optimized)
