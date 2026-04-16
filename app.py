import gradio as gr
from predict import predict_ticket
from bert_predict import predict_bert

def classify_ticket(text):
    category, priority = predict_ticket(text)
    bert_category = predict_bert(text)
    
    return f"""
       **Category:** {category}\n
       **Priority:** {priority}\n
       **BERT Category:** {bert_category}
    """
    
interface = gr.Interface(
    fn=classify_ticket,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Describe your issue here..."
    ),
    outputs=gr.Markdown( ),
    title="AI Customer Support Classifier",
    description="Compares ML model vs BERT model"
)

interface.launch()