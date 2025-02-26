from utils import State, retrieve, generate
from langgraph.graph import START, StateGraph
import gradio as gr


# Define RAG Graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def rag_interface(query):
    result = graph.invoke({"question": query})
    return result["answer"]

with gr.Blocks() as demo:
    gr.Markdown("# RAG-based Question Answering")
    with gr.Row():
        query_input = gr.Textbox(label="Enter your query:")
        submit_button = gr.Button("Ask")
    response_output = gr.Textbox(label="Response", interactive=False)
    
    submit_button.click(rag_interface, inputs=[query_input], outputs=[response_output])

demo.launch()