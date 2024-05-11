# LLM-with-RAG-full-opensource

This code enables you to utilize opensource LLMs locally with the RAG function. You can feed the LLM model with the data from a URL or PDF, resulting in improved answers as the LLM's inference is mostly based on the supplied data.

It wasn't easy to find a free reliable tutorial on this topic, especially when I wanted to use only opensource and run it locally, so that's why I decided to share this code. Finally, I found a super effective sollution here: https://www.youtube.com/watch?v=jENqvjpkwmw&list=LL&index=3. My code is essentially the same at its core. However, I made some small tweaks here and there, modified the Gradio part a bit, and added the Flask part. 
<br><br>

**General step by step tutorial:**
1. Download Ollama (Ollama allows you to run open-source large language models locally) from https://ollama.com/, and install it.
2. Go to https://ollama.com/models, and find the name of a model you want to try, and which your hardware will be able to handle. "mistral" 7b or "llama3" are always a good deafault bets. "mwiewior/bielik" for polish language only.
3. Download a model by openning command line and typing: Ollama pull mistral. Or swap mistral for a model name of your choice.
4. Install needed packages.
<br><br>

**Querying URL with/without RAG:**\
&emsp;5. Just follow the code.

**Querying PDF:**\
&emsp;5. Just follow the code. Put pdf file in your project folder, or supply adequate path to it for PyPDFLoader.

**Using Gradio:**\
&emsp;5. Just follow the code. Run the code, and put http://127.0.0.1:7860 in browser to test it locally, or add in .launch(share=True) to share the generated link with others.

**Using Flask:**\
&emsp;5. Create in your project folder, a folder for temporarily storing PDFs.\
&emsp;6. Create in your project folder, a folder named templates.\
&emsp;7. In templates folder create a file named index3.html and setup everything there or just download mine. A few hints about customizations are even there.\
&emsp;8. Just follow the code.\
&emsp;9. Run the code, go to http://127.0.0.1:5000, and test it.\

**Problems to solve in the future:**
 - It's very slow for big PDF files (around few minutes)
 - It doesn't handle well PDFs with a lot of tables, pictures and strange objects.
 - After few questions usually the restart is needed.
