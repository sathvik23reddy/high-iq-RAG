This is a Retrieval Augmented Generation (RAG) built with Slack Integration so it can answer any queries coming in from a slack chat/channel.

-It was built to work against a documentation of the product to help answer queries related to the product.
-The Model has CodeLlama as the underlying LLM
-QDrant Vector DB to store transformed vectors
-Sentence Transformer as underlying embedding model
-RAGAS as the evaluation pipeline
-Flask cover to expose the API as a service
-ngrok to expose URL to the public
-LangChain and LLamaIndex as frameworks