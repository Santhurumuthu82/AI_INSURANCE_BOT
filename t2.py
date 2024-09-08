import thirdai
import streamlit as st
import nltk
import os
import time
from thirdai import licensing, neural_db as ndb
from dotenv import load_dotenv
load_dotenv()
#nltk.download("punkt")

if os.getenv('THIRD_AI_KEY') :
    licensing.activate(os.getenv('THIRD_AI_KEY'))
db = ndb.NeuralDB()
insertable_docs = []
#streamlit run D.py
#doc_files = [r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-little-champ-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\Technology_Trends_Outlook_2024.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\wealth-maximizer-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\tulip-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\smart-save-plan-brochure (1).pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\single-premium-brochure (1) (1).pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-simple-benefit-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-pos-cash-back-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-money-balance-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-maha-jeevan-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-smart-pay-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-saral-jeevan-bima-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-saral-bachat-bima-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-radiance-smart-investment-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-micro-bachat-plan-brochure (1).pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-long-guaranteed-income-plan-brochure (1) (1).pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-insurance-khata-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-guaranteed-benefit-plan-brochure1 (1).pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-life-elite-term-plan-brochure (1).pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\indiafirst-csc-shubhlabh-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\guaranteed-protection-plus-plan-brochure.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\gold-brochure (1) (2).pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\cash-back-plan-brochuree.pdf", r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\accidental-death-benefit-rider-brochure.pdf"]
doc_files = ["Policies.pdf","accidental-death-benefit-rider-brochure.pdf","cash-back-plan-brochuree.pdf","gold-brochure (1).pdf","gold-brochure.pdf","guaranteed-protection-plus-plan-brochure.pdf","indiafirst-csc-shubhlabh-plan-brochure.pdf","indiafirst-life-elite-term-plan-brochure.pdf","indiafirst-life-guaranteed-benefit-plan-brochure1 (1).pdf","indiafirst-life-guaranteed-benefit-plan-brochure1.pdf","indiafirst-life-long-guaranteed-income-plan-brochure (1).pdf","indiafirst-life-little-champ-plan-brochure.pdf","indiafirst-life-insurance-khata-plan-brochure.pdf","indiafirst-life-micro-bachat-plan-brochure.pdf","indiafirst-life-micro-bachat-plan-brochure (1).pdf","indiafirst-life-long-guaranteed-income-plan-brochure.pdf","indiafirst-life-saral-bachat-bima-plan-brochure.pdf","indiafirst-life-radiance-smart-investment-plan-brochure.pdf","indiafirst-life-plan-brochure.pdf","indiafirst-maha-jeevan-plan-brochure.pdf","indiafirst-life-smart-pay-plan-brochure.pdf","indiafirst-life-saral-jeevan-bima-brochure.pdf","indiafirst-pos-cash-back-plan-brochure.pdf","indiafirst-money-balance-plan-brochure.pdf","indiafirst-maha-jeevan-plan-brochure.pdf","single-premium-brochure.pdf","single-premium-brochure (1).pdf","indiafirst-simple-benefit-plan-brochure.pdf","wealth-maximizer-brochure.pdf","tulip-brochure.pdf","smart-save-plan-brochure.pdf"]
#doc_files = ["C:\\Users\\kamal\\OneDrive\\Desktop\\InsuranceBot\\Policies -  1.CSC Shubhlabh Plan.csv"]
for file in doc_files:
    doc = ndb.PDF(file)
    insertable_docs.append(doc)
db.insert(insertable_docs, train=False)

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] =os.getenv('OPEN_AI_KEY')

from openai import OpenAI
def generate_answers(query, references):
    openai_client = OpenAI()
    context = "\n\n".join(references[:3])

    #prompt = f"As an insurance expert, provide a direct and concise answer to the following question, focusing on any numerical or quantitative aspects first. Use the provided context and explain only the most relevant terms if necessary:\n\nQuestion: {query}\n\nContext: {context}\n\nPlease avoid asking for additional information unless absolutely necessary, and give a clear answer based on the available context."
    prompt = (
        """You are an advanced insurance agent chatbot integrated with NeuralDB and OpenAI. Your role is to assist users by answering their insurance-related queries based on the 29 uploaded policy documents. You must deliver highly accurate, clear, and relevant information to the user's questions. Here’s how you should interact:

    Greeting: Always begin with a warm, professional greeting when the user starts the conversation or greets you. Acknowledge each greeting in a manner that fits the flow of the conversation. For example, "Hello! How can I assist you with your insurance queries today?" or "Hi there! What insurance details would you like to know more about?"

    Context Awareness & Memory: Remember the entire conversation history within the current session to deliver responses based on prior exchanges. Reference the user’s previous questions and comments to create a personalized experience. For example: "Earlier, you mentioned interest in health insurance. Let me expand on that for you."

    Accuracy & Policy Referencing:

    Answer questions using precise information from the 29 uploaded insurance policy documents.
    Directly pull relevant clauses, definitions, coverage details, exclusions, and conditions from these documents to ensure responses are factually correct.
    Always provide answers in an easy-to-understand format, even when the policy wording is complex.
    Clarity: When responding, always be clear, concise, and ensure your explanation is tailored to the user’s specific question. If the user is unclear, proactively clarify or offer additional relevant information.

    Handling Follow-ups & Repeated Questions:

    If the user asks for further details or repeats a query, provide expanded information while linking it to previous answers.
    Maintain professionalism in all follow-up interactions, ensuring you continuously provide value in each response.
    Error Handling: If you do not find the requested information in the uploaded documents, politely inform the user and ask for more context, or offer to explain related topics.

    Response Format:

    Greeting & Engagement: Acknowledge user greetings and keep the tone friendly.
    Accurate Information: Always reference the correct policy information based on user queries.
    Memory: Tailor responses based on previous questions and provide a seamless experience.
    Example Interactions:

    User: "Hi" Chatbot: "Hello! How can I assist you with your insurance queries today?"

    User: "What does this health insurance cover?" Chatbot: "Great question! According to the policy documents, this health insurance covers hospital expenses, doctor consultations, and specific treatments like surgeries. It also includes coverage for pre-existing conditions after a waiting period of 2 years. Would you like more details on exclusions or premiums?"

    User: "Can you tell me more about exclusions?" Chatbot: "Certainly! Exclusions for this policy include cosmetic surgeries, non-prescription drugs, and any treatments deemed experimental. Pre-existing conditions are excluded for the first two years. Do you need more information on specific exclusions?"
    you only refer the this docs:{doc_files}  
    """  
    f"Question: {query}\n"
    f"Context: {context}\n"
    
    "You have to follow the instructions which are enclosed in triple quotes and provide only the answer to the user.\n"
    "ask if there is any additional doubt on the provided answer to the user.\n"
    )   
    messages = [{"role": "user", "content": prompt}]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0
    )
    return response.choices[0].message.content

def generate_queries_chatgpt(original_query):
    openai_client = OpenAI()
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (5 queries):"}
        ]
    )

    generated_queries = response.choices[0].message.content.strip().split("\n")
    return generated_queries

def get_references(query):
    search_results = db.search(query, top_k=100)
    references = []
    for result in search_results:
        references.append(result.text)
    return references

def reciprocal_rank_fusion(reference_list, k=60):
    fused_scores = {}
        
    for i in reference_list:
        for rank, j in enumerate(i):
            if j not in fused_scores:
                fused_scores[j] = 0
            fused_scores[j] += 1 / ((rank+1) + k)
    
    reranked_results = {}
    sorted_fused_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    for j, score in sorted_fused_scores:
        reranked_results[j] = score
    return reranked_results

def get_answer(query, r):
    return generate_answers(
        query=query,
        references=r
    )

st.set_page_config(page_title="Insurance Bot", page_icon=":robot_face:", layout="centered")

def main():
    st.title("Insurance Bot 🤖")
    st.write("Who summon me here!")
    st.write("Is it You, What you want to know?🧙‍♀️ ")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    chat_placeholder = st.empty()

    display_chat_history(chat_placeholder)

    query = st.chat_input("Enter your question...", key="unique_query_input")

    if query:
        st.session_state["chat_history"].append({"user": query, "bot": "..."})

        display_chat_history(chat_placeholder)

        #with st.spinner("Bot is typing..."):
        query_list = generate_queries_chatgpt(query)
        reference_list = [get_references(q) for q in query_list]
        r = reciprocal_rank_fusion(reference_list)
        ranked_reference_list = list(r.keys())
        ans = get_answer(query, ranked_reference_list)

        st.session_state["chat_history"][-1]["bot"] = ans

        display_chat_history(chat_placeholder)

def display_chat_history(placeholder):
    with placeholder.container():
        for chat in st.session_state["chat_history"]:
            st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                    <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; max-width: 60%;">
                        <strong>You:</strong> {chat['user']} <span style="font-size: 20px;">👤</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if chat['bot']:
                st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                        <div style="background-color: #e0f7fa; padding: 10px; border-radius: 5px; max-width: 60%;">
                            <span style="font-size: 20px;">🤖</span> <strong>Bot:</strong> {chat['bot']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()




