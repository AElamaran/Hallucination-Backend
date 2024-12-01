from langchain_aws import BedrockLLM

mitsral_large= "mistral.mistral-large-2402-v1:0"
meta_llama3_70b="meta.llama3-70b-instruct-v1:0"

def get_bedrock_model():
    # Initialize the Bedrock model
    bedrock_client = BedrockLLM(
        model_id= meta_llama3_70b,
        region_name="us-east-1",  # Choose the appropriate region
        credentials_profile_name="default"  # Use your default AWS profile or specify a different one
    )

    # Create a LangChain LLMChain
    return bedrock_client

