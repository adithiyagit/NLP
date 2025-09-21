from flask import Flask, render_template, request, session
from models import generate_response

app = Flask(__name__)
app.secret_key = "adhi-secret-key"

@app.route("/", methods=["GET", "POST"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_input = request.form["user_input"]
        session["chat_history"].append({"sender": "user", "message": user_input})

        # Summarize input text
        response = generate_response(user_input, session["chat_history"])
        session["chat_history"].append({"sender": "assistant", "message": response})
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

if __name__ == "__main__":
    app.run(debug=True)
