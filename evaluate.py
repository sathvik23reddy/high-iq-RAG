from ragas import SingleTurnSample
from ragas.metrics import BleuScore
import rag

metric = BleuScore()

#Range as per automation
for i in range(0, 10):
    #Get from automation
    user_input, reference = 0, 0

    response = rag.init(user_input)

    sample = SingleTurnSample(
        user_input=user_input,
        response=response,
        reference=reference,
    )

    result = metric.single_turn_score(sample)

    #Output res into automation
    print(result)

