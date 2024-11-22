import fitz
import re
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, TransformersReader
from haystack.pipelines import Pipeline
def text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text =  ""
    for page in doc:
        text += page.get_text()
    return text

pdf_path_here = "thv.pdf"
def split_into_lines(text):
    """Split text into lines."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines
text_extracted = text_from_pdf(pdf_path_here)
print(text_extracted)
def process_text(raw_text):
    structured_data = {
        "objectives": [],
        "progress": [],
        "team_members": []
    }

    current_sec = None
    lines = split_into_lines(raw_text)
    for line in lines:

        if re.match(r'Objectives:', line, re.IGNORECASE):
            current_sec = 'objectives'
        elif re.match(r'Progress', line, re.IGNORECASE):
            current_sec = 'progress'
        elif re.match(r'exitTeam Members', line, re.IGNORECASE):
            current_sec = 'team_members'
        elif current_sec == 'objectives':
            if line:
                structured_data['objectives'].append(line)
        elif current_sec == 'progress':
            if line:
                structured_data['progress'].append(line)
        elif current_sec == 'team_members':
            if line:
                structured_data['team_members'].append(line)
    structured_data['objectives'] = ', '.join(structured_data['objectives'])
    structured_data['progress'] = ', '.join(structured_data['progress'])
    structured_data['team_members'] = ', '.join(structured_data['team_members'])

    return structured_data

processed_text = process_text(text_extracted)
print(processed_text)        
document_store = InMemoryDocumentStore()

documents = [
    Document(content=processed_text['objectives'], meta={"section": "objectives"}),
    Document(content=processed_text['progress'], meta={"section": "progress"}),
    Document(content=processed_text['team_members'], meta={"section": "team_members"}),
]

# Write documents to the document store
document_store.write_documents(documents)

# Step 4: Initialize the retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",  # Choose your model
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False  #set tru if you have gpu
)

# Update embeddings in the document store
document_store.update_embeddings(retriever)

# Step 5: Initialize the generator
reader = TransformersReader(model_name_or_path="facebook/bart-large")
# Step 6: Create the RAG pipeline
pipeline = Pipeline()
pipeline.add_node(retriever, "Retriever", inputs=["Query"])
pipeline.add_node(reader, "Reader", inputs=["Retriever"])
# Step 7: Terminal-based chatbot logic
def chatbot():
    print("\nWelcome to the IIT Bombay Humanoid Project Chatbot!")
    print("Ask your questions or type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")

        if user_query.lower() == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break

        try:
            result = pipeline.run(query=user_query, params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}})
            answer = result['answers'][0].answer
            print("Chatbot:", answer)
        except Exception as e:
            print("Chatbot: Sorry, I couldn't process your query. Please try again.")
            print("Error:", e)

# Step 8: Run the chatbot
if __name__ == "__main__":
    chatbot()