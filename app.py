import os
from flask import Flask, request, render_template
from prompts import transform_chain

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        requested_style = request.form.get("requested_style", "Make it more casual")
        if input_text.strip():
            result = transform_chain.run(input_text=input_text, requested_style=requested_style)
            # result is a dictionary with summary, sentiment, and final_text
            summary = result["summary"]
            sentiment = result["sentiment"]
            final_text = result["final_text"]

            return render_template("index.html", 
                                   input_text=input_text,
                                   requested_style=requested_style,
                                   summary=summary,
                                   sentiment=sentiment,
                                   final_text=final_text)
    return render_template("index.html")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment
    app.run(debug=True)
