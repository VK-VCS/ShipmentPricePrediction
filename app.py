from flask import Flask, jsonify,request
from PricingModel import PricingModel

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    msg = ""
    if request.method == "POST":
        obj = PricingModel(request.json)
        msg = obj.check_mandatory_fields()
        if msg == "":
            msg = str(obj.predict())

    return jsonify(msg)


if __name__ == '__main__':
    app.run()
