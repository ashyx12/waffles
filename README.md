# Waffles: Educational Chatbot

## Description

Waffles is an intelligent question-answering system designed to help you understand complex topics by allowing you to "chat" with your documents. The goal of this project is to create a versatile tool that can be used across **all subjects**, from computer science to history.

This version is the first step toward that goal, with a focus on **Operating Systems**. If you've ever struggled to find a specific piece of information in a dense PowerPoint presentation or a lengthy textbook, Waffles is here to help.

## LIVE DEMO

You can check out a live version of the website here: **[https://waffles-glm9.onrender.com]**

## Features

- **Natural Language Questions:** Ask questions in plain English, just like you would talk to a person.
- **Powered by Gemini:** Utilizes Google's powerful Gemini Pro model to provide clear and elaborate answers.
- **Source Verification:** Every answer is accompanied by the names of the source documents.

## Future Scope

- **Multi-Subject Support:** Implementing a system to manage multiple, independent knowledge bases for different subjects.
- **User Interface Enhancements:** Adding features to allow users to select which subject they want to query.
- **Broader File Type Support:** Expanding the range of document types that can be indexed.

## How It Works

Waffles uses a technique called **Retrieval-Augmented Generation (RAG)**.

1.  **Indexing:**
    - The application reads your documents from a local directory.
    - It splits them into smaller chunks and converts them into numerical vectors using **GoogleGenerativeAIEmbeddings** model.
    - These vectors are stored in a local **FAISS vector database**.

2.  **Querying:**
    - When you ask a question, it is converted into a vector.
    - The system searches the database for the most relevant text chunks.
    - These chunks and your original question are sent to the **Gemini Pro** model, which generates a detailed answer.

## Getting Started

### Prerequisites

- Python 3.8+
- A Google API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ashyx12/waffles.git
    cd Waffles
    ```
   
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
   
3.  **Set up your environment variables:**
    Create a `.env` file and add your Google API key:
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```
   
4.  **Add Your Documents:**
    For now, place your documents into the `data/ES` directory.

5.  **Create the Vector Database:**
    Run the `main.py` script to process your documents.
    ```bash
    python main.py
    ```
6. **Ask Questions:**
   You can ask questions from the documents you have uploaded
